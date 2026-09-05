import inspect
import json
import typing

import pytest

from pairedrl.env.backoffice import (
    CONTROL_TOOLS,
    PAGE_SIZE,
    READ_TOOLS,
    TOOL_NAMES,
    WRITE_TOOLS,
    ToolAPI,
    World,
    order_fully_reserved,
    order_total_cents,
)


@pytest.fixture
def api():
    return ToolAPI(World.generate(11))


def call(api, name, **kwargs):
    return json.loads(getattr(api, name)(**kwargs))


def first_order(api, **conditions):
    for o in sorted(api.world.orders.values(), key=lambda o: o.order_id):
        if all(getattr(o, k) == v for k, v in conditions.items()):
            return o
    raise AssertionError(f"no order with {conditions}")


def public_tool_names():
    return {
        name
        for name, member in inspect.getmembers(ToolAPI, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def test_public_methods_are_exactly_the_tools():
    assert public_tool_names() == set(TOOL_NAMES)
    assert READ_TOOLS | WRITE_TOOLS | CONTROL_TOOLS == TOOL_NAMES
    assert not (READ_TOOLS & WRITE_TOOLS) and not (READ_TOOLS & CONTROL_TOOLS)
    assert len(TOOL_NAMES) == 13


def test_every_tool_documents_every_argument():
    for name in TOOL_NAMES:
        fn = getattr(ToolAPI, name)
        doc = inspect.getdoc(fn)
        assert doc and "Args:" in doc, name
        hints = typing.get_type_hints(fn)
        for param in list(inspect.signature(fn).parameters)[1:]:
            assert f"{param}:" in doc, f"{name} lacks docs for {param}"
            assert hints[param] in (str, int), f"{name}.{param} has a non-JSON type"
        assert hints["return"] is str


def test_every_tool_returns_json(api):
    for name in TOOL_NAMES:
        fn = getattr(api, name)
        kwargs = {p: "" for p in list(inspect.signature(fn).parameters)}
        json.loads(fn(**kwargs))


def test_search_paginates_and_validates(api):
    page = call(api, "search_customers", query="a")
    assert page["total"] >= PAGE_SIZE and len(page["results"]) == PAGE_SIZE
    assert page["offset"] == 0 and page["next_offset"] == PAGE_SIZE
    page2 = call(api, "search_customers", query="a", offset=page["next_offset"])
    assert page2["offset"] == PAGE_SIZE
    assert call(api, "search_customers", query="")["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "search_customers", query="a", offset=-1)["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "search_customers", query="zzzzzz")["total"] == 0


def test_search_matches_name_email_and_phone(api):
    c = api.world.customers["C0003"]
    assert any(r["customer_id"] == "C0003" for r in call(api, "search_customers", query=c.name)["results"])
    assert any(r["customer_id"] == "C0003" for r in call(api, "search_customers", query=c.email)["results"])
    assert any(r["customer_id"] == "C0003" for r in call(api, "search_customers", query=c.phone[-6:])["results"])


def test_get_customer(api):
    rec = call(api, "get_customer", customer_id="C0001")
    assert rec["customer_id"] == "C0001" and set(rec["address"]) == {"street", "city", "postal_code"}
    assert call(api, "get_customer", customer_id="C9999")["error"]["code"] == "NOT_FOUND"


def test_list_orders_and_get_order(api):
    cid = first_order(api).customer_id
    page = call(api, "list_orders", customer_id=cid)
    assert page["total"] >= 1 and all(r["status"] for r in page["results"])
    ids = [r["order_id"] for r in page["results"]]
    assert ids == sorted(ids, reverse=True)
    order = call(api, "get_order", order_id=ids[0])
    assert order["total_cents"] == sum(i["quantity"] * i["unit_price_cents"] for i in order["items"])
    assert "refunded_cents" in order and "reservations" in order
    assert call(api, "list_orders", customer_id="C9999")["error"]["code"] == "NOT_FOUND"
    assert call(api, "get_order", order_id="O9999")["error"]["code"] == "NOT_FOUND"


def test_check_inventory(api):
    inv = call(api, "check_inventory", sku="SKU-0001")
    assert inv["available"] == inv["stock"] - inv["reserved"]
    assert call(api, "check_inventory", sku="SKU-9999")["error"]["code"] == "NOT_FOUND"


def test_update_address_rules(api):
    pending = first_order(api, status="pending")
    out = call(api, "update_shipping_address", order_id=pending.order_id, street="1 New St", city="Dhaka", postal_code="1200")
    assert out["shipping_address"]["city"] == "Dhaka" and pending.shipping_address.street == "1 New St"
    shipped = first_order(api, status="shipped")
    err = call(api, "update_shipping_address", order_id=shipped.order_id, street="1 New St", city="Dhaka", postal_code="1200")
    assert err["error"]["code"] == "INVALID_STATE"
    err = call(api, "update_shipping_address", order_id=pending.order_id, street="", city="Dhaka", postal_code="1200")
    assert err["error"]["code"] == "INVALID_ARGUMENT"
    out = call(api, "update_shipping_address", order_id=pending.order_id, street="2 New St", city="Sylhet", postal_code=3100)
    assert out["shipping_address"]["postal_code"] == "3100" and pending.shipping_address.postal_code == "3100"
    err = call(api, "update_shipping_address", order_id=pending.order_id, street="2 New St", city="Sylhet", postal_code=True)
    assert err["error"]["code"] == "INVALID_ARGUMENT"
    assert json.loads(api.get_order("nope"))["error"]["code"] == "NOT_FOUND"
    api.finish("done")
    assert "do not call any more tools" in json.loads(api.get_order(pending.order_id))["error"]["message"]


def test_cancel_rules(api):
    paid = first_order(api, status="paid")
    before = api.world.version
    out = call(api, "cancel_order", order_id=paid.order_id, reason="customer request")
    assert out["status"] == "cancelled" and out["refund_due_cents"] == order_total_cents(paid)
    assert paid.paid_before_cancel and paid.reservations == [] and api.world.version == before + 1
    pending = first_order(api, status="pending")
    assert call(api, "cancel_order", order_id=pending.order_id, reason="dup")["refund_due_cents"] == 0
    shipped = first_order(api, status="shipped")
    assert call(api, "cancel_order", order_id=shipped.order_id, reason="x")["error"]["code"] == "INVALID_STATE"
    assert call(api, "cancel_order", order_id=paid.order_id, reason="")["error"]["code"] == "INVALID_ARGUMENT"


def test_refund_rules(api):
    pending = first_order(api, status="pending")
    assert call(api, "issue_refund", order_id=pending.order_id, amount_cents=100, reason="r")["error"]["code"] == "RULE_VIOLATION"
    shipped = first_order(api, status="shipped")
    assert call(api, "issue_refund", order_id=shipped.order_id, amount_cents=100, reason="r")["error"]["code"] == "RULE_VIOLATION"
    delivered = first_order(api, status="delivered")
    balance = order_total_cents(delivered) - sum(r.amount_cents for r in delivered.refunds)
    err = call(api, "issue_refund", order_id=delivered.order_id, amount_cents=balance + 1, reason="r")
    assert err["error"]["code"] == "RULE_VIOLATION" and err["error"]["refundable_balance_cents"] == balance
    ok = call(api, "issue_refund", order_id=delivered.order_id, amount_cents=balance, reason="damaged")
    assert ok["refundable_balance_cents"] == 0
    assert call(api, "issue_refund", order_id=delivered.order_id, amount_cents=1, reason="r")["error"]["code"] == "RULE_VIOLATION"
    assert call(api, "issue_refund", order_id=delivered.order_id, amount_cents=0, reason="r")["error"]["code"] == "INVALID_ARGUMENT"


def test_refund_after_cancel_depends_on_payment(api):
    paid = first_order(api, status="paid")
    call(api, "cancel_order", order_id=paid.order_id, reason="customer request")
    assert call(api, "issue_refund", order_id=paid.order_id, amount_cents=order_total_cents(paid), reason="cancel")["refundable_balance_cents"] == 0
    pending = first_order(api, status="pending")
    call(api, "cancel_order", order_id=pending.order_id, reason="dup")
    assert call(api, "issue_refund", order_id=pending.order_id, amount_cents=1, reason="r")["error"]["code"] == "RULE_VIOLATION"


def test_reserve_rules(api):
    order = next(
        o for o in sorted(api.world.orders.values(), key=lambda o: o.order_id)
        if o.status == "paid" and not o.reservations
    )
    item = order.items[0]
    api.world.products[item.sku].stock = item.quantity + 5
    other_sku = next(s for s in api.world.products if all(i.sku != s for i in order.items))
    assert call(api, "reserve_stock", sku=other_sku, quantity=1, order_id=order.order_id)["error"]["code"] == "RULE_VIOLATION"
    assert call(api, "reserve_stock", sku=item.sku, quantity=item.quantity + 1, order_id=order.order_id)["error"]["code"] == "RULE_VIOLATION"
    assert call(api, "reserve_stock", sku=item.sku, quantity=0, order_id=order.order_id)["error"]["code"] == "INVALID_ARGUMENT"
    ok = call(api, "reserve_stock", sku=item.sku, quantity=item.quantity, order_id=order.order_id)
    assert ok["reserved_quantity"] == item.quantity
    api.world.products[item.sku].stock = 0
    order2 = next(
        o for o in sorted(api.world.orders.values(), key=lambda o: o.order_id)
        if o.status == "pending" and any(i.sku == item.sku for i in o.items) and o.order_id != order.order_id
    ) if any(
        o.status == "pending" and any(i.sku == item.sku for i in o.items) and o.order_id != order.order_id
        for o in api.world.orders.values()
    ) else None
    if order2 is not None:
        err = call(api, "reserve_stock", sku=item.sku, quantity=1, order_id=order2.order_id)
        assert err["error"]["code"] == "RULE_VIOLATION" and "available" in err["error"]
    shipped = first_order(api, status="shipped")
    assert call(api, "reserve_stock", sku=shipped.items[0].sku, quantity=1, order_id=shipped.order_id)["error"]["code"] == "INVALID_STATE"


def test_schedule_shipment_rules(api):
    today = api.world.today_day
    pending = first_order(api, status="pending")
    assert call(api, "schedule_shipment", order_id=pending.order_id, carrier="standard", day=today)["error"]["code"] == "INVALID_STATE"
    reserved = next(o for o in sorted(api.world.orders.values(), key=lambda o: o.order_id) if o.status == "paid" and order_fully_reserved(o))
    unreserved = next(o for o in sorted(api.world.orders.values(), key=lambda o: o.order_id) if o.status == "paid" and not order_fully_reserved(o))
    assert call(api, "schedule_shipment", order_id=unreserved.order_id, carrier="standard", day=today)["error"]["code"] == "RULE_VIOLATION"
    assert call(api, "schedule_shipment", order_id=reserved.order_id, carrier="pigeon", day=today)["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "schedule_shipment", order_id=reserved.order_id, carrier="standard", day=today + 8)["error"]["code"] == "RULE_VIOLATION"
    assert call(api, "schedule_shipment", order_id=reserved.order_id, carrier="standard", day=today - 1)["error"]["code"] == "RULE_VIOLATION"
    stock_before = {i.sku: api.world.products[i.sku].stock for i in reserved.items}
    out = call(api, "schedule_shipment", order_id=reserved.order_id, carrier="Express", day=today + 2)
    assert out["status"] == "scheduled" and out["carrier"] == "express"
    assert reserved.status == "shipped" and reserved.shipment_id == out["shipment_id"] and reserved.reservations == []
    for i in reserved.items:
        assert api.world.products[i.sku].stock == stock_before[i.sku] - i.quantity
    assert api.world.shipments[out["shipment_id"]].scheduled_day == today + 2
    assert call(api, "schedule_shipment", order_id=reserved.order_id, carrier="standard", day=today)["error"]["code"] == "INVALID_STATE"


def test_create_ticket_rules(api):
    order = first_order(api)
    other = next(o for o in api.world.orders.values() if o.customer_id != order.customer_id)
    out = call(api, "create_ticket", customer_id=order.customer_id, order_id=order.order_id, category="Shipping", priority="high", summary="Late")
    assert out["status"] == "open" and out["category"] == "shipping" and out["ticket_id"] in api.world.tickets
    out2 = call(api, "create_ticket", customer_id=order.customer_id, order_id="", category="account", priority="low", summary="Login")
    assert out2["order_id"] is None
    assert call(api, "create_ticket", customer_id=order.customer_id, order_id=other.order_id, category="billing", priority="low", summary="x")["error"]["code"] == "RULE_VIOLATION"
    assert call(api, "create_ticket", customer_id=order.customer_id, order_id="", category="weird", priority="low", summary="x")["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "create_ticket", customer_id=order.customer_id, order_id="", category="billing", priority="urgent", summary="x")["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "create_ticket", customer_id="C9999", order_id="", category="billing", priority="low", summary="x")["error"]["code"] == "NOT_FOUND"


def test_wait_and_finish(api):
    assert call(api, "wait", seconds=30)["clock_seconds"] == 30
    assert call(api, "wait", seconds=0)["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "wait", seconds=601)["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "finish", summary="")["error"]["code"] == "INVALID_ARGUMENT"
    assert call(api, "finish", summary="done")["status"] == "finished"
    assert api.finished and api.finish_summary == "done"
    assert call(api, "get_customer", customer_id="C0001")["error"]["code"] == "EPISODE_FINISHED"
    assert call(api, "finish", summary="again")["error"]["code"] == "EPISODE_FINISHED"


def test_reads_do_not_commit_and_writes_do(api):
    v = api.world.version
    call(api, "get_customer", customer_id="C0001")
    call(api, "search_customers", query="a")
    call(api, "wait", seconds=5)
    assert api.world.version == v
    pending = first_order(api, status="pending")
    call(api, "update_shipping_address", order_id=pending.order_id, street="2 St", city="Sylhet", postal_code="3100")
    assert api.world.version == v + 1


def test_call_log_records_outcomes(api):
    call(api, "get_customer", customer_id="C0001")
    call(api, "get_customer", customer_id="C9999")
    assert [e["ok"] for e in api.call_log] == [True, False]
    assert api.call_log[1]["code"] == "NOT_FOUND" and api.call_log[0]["tool"] == "get_customer"


def test_integer_arguments_accept_numeric_strings(api):
    page = call(api, "search_customers", query="a", offset="5")
    assert page["offset"] == 5
