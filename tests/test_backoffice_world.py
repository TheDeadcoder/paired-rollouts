import json

from pairedrl.env.backoffice import World, order_fully_reserved, order_total_cents


def test_generation_is_deterministic():
    a = World.generate(7).snapshot()
    b = World.generate(7).snapshot()
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_different_seeds_differ():
    assert World.generate(7).snapshot() != World.generate(8).snapshot()


def test_counts_and_ids():
    w = World.generate(1)
    assert len(w.customers) == 40 and len(w.products) == 30 and len(w.orders) == 80
    assert len(w.tickets) == 10
    assert all(k == v.customer_id for k, v in w.customers.items())
    assert all(k == v.order_id for k, v in w.orders.items())
    assert all(k == v.shipment_id for k, v in w.shipments.items())
    assert len({c.name for c in w.customers.values()}) == 40


def test_referential_integrity():
    w = World.generate(2)
    for o in w.orders.values():
        assert o.customer_id in w.customers
        assert o.items and all(i.sku in w.products for i in o.items)
        if o.status in ("shipped", "delivered"):
            assert o.shipment_id in w.shipments
            assert w.shipments[o.shipment_id].order_id == o.order_id
        else:
            assert o.shipment_id is None
        for r in o.reservations:
            assert any(i.sku == r.sku for i in o.items)
    for t in w.tickets.values():
        assert t.customer_id in w.customers
        assert t.order_id is None or w.orders[t.order_id].customer_id == t.customer_id


def test_stock_covers_reservations_and_totals_positive():
    w = World.generate(3)
    for sku in w.products:
        assert w.available(sku) >= 0
    assert all(order_total_cents(o) > 0 for o in w.orders.values())
    assert any(o.status == "paid" and order_fully_reserved(o) for o in w.orders.values())
    assert any(o.status == "paid" and not order_fully_reserved(o) for o in w.orders.values())


def test_status_mix_present():
    w = World.generate(4)
    statuses = {o.status for o in w.orders.values()}
    assert statuses == {"pending", "paid", "shipped", "delivered", "cancelled"}


def test_commit_and_as_of():
    w = World.generate(5)
    assert w.version == 0 and len(w.history) == 1
    order = next(o for o in w.orders.values() if o.status == "pending")
    order.status = "cancelled"
    assert w.commit() == 1
    assert len(w.history) == 2
    assert w.as_of(0)["orders"][order.order_id]["status"] == "pending"
    assert w.as_of(1)["orders"][order.order_id]["status"] == "cancelled"
    assert w.as_of(-5)["version"] == 0
    assert w.as_of(99)["version"] == 1


def test_snapshot_is_json_serializable_and_sorted():
    snap = World.generate(6).snapshot()
    text = json.dumps(snap)
    assert '"customers"' in text
    assert list(snap["orders"]) == sorted(snap["orders"])
