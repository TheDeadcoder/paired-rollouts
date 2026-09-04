"""Tool API over the back-office world. Public methods of ToolAPI are the agent's tools."""

import json

from pairedrl.env.backoffice.world import (
    CARRIERS,
    PAGE_SIZE,
    SHIPPING_WINDOW_DAYS,
    TICKET_CATEGORIES,
    TICKET_PRIORITIES,
    Refund,
    Reservation,
    Shipment,
    Ticket,
    World,
    order_fully_reserved,
    order_refunded_cents,
    order_reserved_quantity,
    order_total_cents,
)

READ_TOOLS = frozenset(
    {"search_customers", "get_customer", "list_orders", "get_order", "check_inventory"}
)
WRITE_TOOLS = frozenset(
    {
        "update_shipping_address",
        "cancel_order",
        "issue_refund",
        "reserve_stock",
        "schedule_shipment",
        "create_ticket",
    }
)
CONTROL_TOOLS = frozenset({"wait", "finish"})
TOOL_NAMES = frozenset(READ_TOOLS | WRITE_TOOLS | CONTROL_TOOLS)

MAX_WAIT_SECONDS = 600


class ToolError(Exception):
    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def _dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _error(code: str, message: str, **extra) -> str:
    return _dumps({"error": {"code": code, "message": message, **extra}})


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("INVALID_ARGUMENT", f"{name} must be a non-empty string")
    return value.strip()


def _require_int(value: object, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        else:
            raise ToolError("INVALID_ARGUMENT", f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ToolError("INVALID_ARGUMENT", f"{name} must be at least {minimum}")
    return value


def _page(items: list, offset: int) -> dict:
    total = len(items)
    page = items[offset : offset + PAGE_SIZE]
    next_offset = offset + PAGE_SIZE if offset + PAGE_SIZE < total else None
    return {"results": page, "total": total, "offset": offset, "next_offset": next_offset}


class ToolAPI:
    """Twelve business tools plus finish. Every tool returns a compact JSON string."""

    def __init__(self, world: World):
        self.world = world
        self.finished = False
        self.finish_summary: str | None = None
        self.call_log: list[dict] = []

    def _view(self) -> dict:
        return self.world.snapshot()

    def _run(self, name: str, fn, **kwargs) -> str:
        if self.finished:
            self.call_log.append({"tool": name, "args": kwargs, "ok": False, "code": "EPISODE_FINISHED"})
            return _error("EPISODE_FINISHED", "the episode was already finished")
        try:
            payload = fn(**kwargs)
        except ToolError as e:
            self.call_log.append({"tool": name, "args": kwargs, "ok": False, "code": e.code})
            return _error(e.code, e.message, **e.extra)
        self.call_log.append({"tool": name, "args": kwargs, "ok": True, "code": None})
        return _dumps(payload)

    def _order(self, order_id: str):
        order = self.world.orders.get(order_id)
        if order is None:
            raise ToolError("NOT_FOUND", f"order {order_id} does not exist")
        return order

    def _customer(self, customer_id: str):
        customer = self.world.customers.get(customer_id)
        if customer is None:
            raise ToolError("NOT_FOUND", f"customer {customer_id} does not exist")
        return customer

    def _product(self, sku: str):
        product = self.world.products.get(sku)
        if product is None:
            raise ToolError("NOT_FOUND", f"sku {sku} does not exist")
        return product

    def search_customers(self, query: str, offset: int = 0) -> str:
        """Search customers by a substring of their name, email or phone number.

        Args:
            query: Text to match, case-insensitive, against name, email and phone.
            offset: Zero-based index of the first result to return, for pagination.
        """
        return self._run("search_customers", self._search_customers, query=query, offset=offset)

    def _search_customers(self, query, offset):
        query = _require_text(query, "query").lower()
        offset = _require_int(offset, "offset", minimum=0)
        view = self._view()["customers"]
        hits = [
            {"customer_id": c["customer_id"], "name": c["name"], "email": c["email"], "phone": c["phone"]}
            for _, c in sorted(view.items())
            if query in c["name"].lower() or query in c["email"].lower() or query in c["phone"]
        ]
        return _page(hits, offset)

    def get_customer(self, customer_id: str) -> str:
        """Return a customer's full record, including address and tier.

        Args:
            customer_id: The customer identifier, for example C0007.
        """
        return self._run("get_customer", self._get_customer, customer_id=customer_id)

    def _get_customer(self, customer_id):
        customer_id = _require_text(customer_id, "customer_id")
        view = self._view()["customers"]
        if customer_id not in view:
            raise ToolError("NOT_FOUND", f"customer {customer_id} does not exist")
        return view[customer_id]

    def list_orders(self, customer_id: str, offset: int = 0) -> str:
        """List a customer's orders with status and total, newest order id first.

        Args:
            customer_id: The customer identifier whose orders to list.
            offset: Zero-based index of the first result to return, for pagination.
        """
        return self._run("list_orders", self._list_orders, customer_id=customer_id, offset=offset)

    def _list_orders(self, customer_id, offset):
        customer_id = _require_text(customer_id, "customer_id")
        offset = _require_int(offset, "offset", minimum=0)
        view = self._view()
        if customer_id not in view["customers"]:
            raise ToolError("NOT_FOUND", f"customer {customer_id} does not exist")
        rows = [
            {
                "order_id": o["order_id"],
                "status": o["status"],
                "total_cents": sum(i["quantity"] * i["unit_price_cents"] for i in o["items"]),
                "placed_on_day": o["placed_on_day"],
                "item_count": len(o["items"]),
            }
            for _, o in sorted(view["orders"].items(), reverse=True)
            if o["customer_id"] == customer_id
        ]
        return _page(rows, offset)

    def get_order(self, order_id: str) -> str:
        """Return an order's full record: items, status, address, refunds, reservations, shipment.

        Args:
            order_id: The order identifier, for example O0042.
        """
        return self._run("get_order", self._get_order, order_id=order_id)

    def _get_order(self, order_id):
        order_id = _require_text(order_id, "order_id")
        view = self._view()["orders"]
        if order_id not in view:
            raise ToolError("NOT_FOUND", f"order {order_id} does not exist")
        o = dict(view[order_id])
        o["total_cents"] = sum(i["quantity"] * i["unit_price_cents"] for i in o["items"])
        o["refunded_cents"] = sum(r["amount_cents"] for r in o["refunds"])
        return o

    def check_inventory(self, sku: str) -> str:
        """Return stock, reserved and available quantities for a product.

        Args:
            sku: The product identifier, for example SKU-0012.
        """
        return self._run("check_inventory", self._check_inventory, sku=sku)

    def _check_inventory(self, sku):
        sku = _require_text(sku, "sku")
        view = self._view()
        if sku not in view["products"]:
            raise ToolError("NOT_FOUND", f"sku {sku} does not exist")
        product = view["products"][sku]
        reserved = sum(
            r["quantity"]
            for o in view["orders"].values()
            for r in o["reservations"]
            if r["sku"] == sku
        )
        return {
            "sku": sku,
            "name": product["name"],
            "price_cents": product["price_cents"],
            "stock": product["stock"],
            "reserved": reserved,
            "available": product["stock"] - reserved,
        }

    def update_shipping_address(self, order_id: str, street: str, city: str, postal_code: str) -> str:
        """Change the shipping address of an order that has not shipped yet.

        Args:
            order_id: The order to update.
            street: Street line of the new address.
            city: City of the new address.
            postal_code: Postal code of the new address.
        """
        return self._run(
            "update_shipping_address",
            self._update_shipping_address,
            order_id=order_id,
            street=street,
            city=city,
            postal_code=postal_code,
        )

    def _update_shipping_address(self, order_id, street, city, postal_code):
        order = self._order(_require_text(order_id, "order_id"))
        street = _require_text(street, "street")
        city = _require_text(city, "city")
        postal_code = _require_text(postal_code, "postal_code")
        if order.status not in ("pending", "paid"):
            raise ToolError(
                "INVALID_STATE",
                f"order {order.order_id} is {order.status}; the address can only change before shipment",
            )
        order.shipping_address.street = street
        order.shipping_address.city = city
        order.shipping_address.postal_code = postal_code
        self.world.commit()
        return {
            "order_id": order.order_id,
            "shipping_address": {"street": street, "city": city, "postal_code": postal_code},
        }

    def cancel_order(self, order_id: str, reason: str) -> str:
        """Cancel an order that has not shipped yet and release its stock reservations.

        Args:
            order_id: The order to cancel.
            reason: Short reason for the cancellation.
        """
        return self._run("cancel_order", self._cancel_order, order_id=order_id, reason=reason)

    def _cancel_order(self, order_id, reason):
        order = self._order(_require_text(order_id, "order_id"))
        _require_text(reason, "reason")
        if order.status not in ("pending", "paid"):
            raise ToolError(
                "INVALID_STATE", f"order {order.order_id} is {order.status} and cannot be cancelled"
            )
        was_paid = order.status == "paid"
        order.status = "cancelled"
        order.paid_before_cancel = was_paid
        order.reservations = []
        self.world.commit()
        return {
            "order_id": order.order_id,
            "status": "cancelled",
            "refund_due_cents": order_total_cents(order) - order_refunded_cents(order) if was_paid else 0,
        }

    def issue_refund(self, order_id: str, amount_cents: int, reason: str) -> str:
        """Refund part or all of an order that was paid, delivered, or cancelled after payment.

        Args:
            order_id: The order to refund.
            amount_cents: Refund amount in cents; must not exceed the refundable balance.
            reason: Short reason for the refund.
        """
        return self._run(
            "issue_refund",
            self._issue_refund,
            order_id=order_id,
            amount_cents=amount_cents,
            reason=reason,
        )

    def _issue_refund(self, order_id, amount_cents, reason):
        order = self._order(_require_text(order_id, "order_id"))
        amount_cents = _require_int(amount_cents, "amount_cents", minimum=1)
        reason = _require_text(reason, "reason")
        if order.status == "pending":
            raise ToolError("RULE_VIOLATION", "nothing has been charged on a pending order")
        if order.status == "shipped":
            raise ToolError(
                "RULE_VIOLATION", "refunds are not allowed while the order is in transit; wait for delivery"
            )
        if order.status == "cancelled" and not order.paid_before_cancel:
            raise ToolError("RULE_VIOLATION", "the order was cancelled before payment; nothing to refund")
        balance = order_total_cents(order) - order_refunded_cents(order)
        if amount_cents > balance:
            raise ToolError(
                "RULE_VIOLATION",
                f"amount {amount_cents} exceeds the refundable balance {balance}",
                refundable_balance_cents=balance,
            )
        order.refunds.append(Refund(amount_cents, reason))
        self.world.commit()
        return {
            "order_id": order.order_id,
            "refund_cents": amount_cents,
            "refunded_total_cents": order_refunded_cents(order),
            "refundable_balance_cents": balance - amount_cents,
        }

    def reserve_stock(self, sku: str, quantity: int, order_id: str) -> str:
        """Reserve stock of a product for an item on a pending or paid order.

        Args:
            sku: The product to reserve; it must be one of the order's items.
            quantity: Number of units to reserve; cannot exceed the ordered quantity.
            order_id: The order the reservation belongs to.
        """
        return self._run(
            "reserve_stock", self._reserve_stock, sku=sku, quantity=quantity, order_id=order_id
        )

    def _reserve_stock(self, sku, quantity, order_id):
        product = self._product(_require_text(sku, "sku"))
        order = self._order(_require_text(order_id, "order_id"))
        quantity = _require_int(quantity, "quantity", minimum=1)
        if order.status not in ("pending", "paid"):
            raise ToolError(
                "INVALID_STATE", f"order {order.order_id} is {order.status}; stock cannot be reserved"
            )
        item = next((i for i in order.items if i.sku == product.sku), None)
        if item is None:
            raise ToolError("RULE_VIOLATION", f"sku {product.sku} is not on order {order.order_id}")
        already = order_reserved_quantity(order, product.sku)
        if already + quantity > item.quantity:
            raise ToolError(
                "RULE_VIOLATION",
                f"order {order.order_id} needs {item.quantity} of {product.sku}; {already} already reserved",
            )
        available = self.world.available(product.sku)
        if quantity > available:
            raise ToolError(
                "RULE_VIOLATION",
                f"only {available} of {product.sku} available",
                available=available,
            )
        order.reservations.append(Reservation(product.sku, quantity))
        self.world.commit()
        return {
            "order_id": order.order_id,
            "sku": product.sku,
            "reserved_quantity": order_reserved_quantity(order, product.sku),
            "available_after": self.world.available(product.sku),
        }

    def schedule_shipment(self, order_id: str, carrier: str, day: int) -> str:
        """Schedule shipment of a paid, fully reserved order within the next seven days.

        Args:
            order_id: The paid order to ship.
            carrier: Either "standard" or "express".
            day: Day number to ship on; today's day number is shown in the task and the window is seven days.
        """
        return self._run(
            "schedule_shipment", self._schedule_shipment, order_id=order_id, carrier=carrier, day=day
        )

    def _schedule_shipment(self, order_id, carrier, day):
        order = self._order(_require_text(order_id, "order_id"))
        carrier = _require_text(carrier, "carrier").lower()
        day = _require_int(day, "day")
        if carrier not in CARRIERS:
            raise ToolError("INVALID_ARGUMENT", f"carrier must be one of {list(CARRIERS)}")
        today = self.world.today_day
        if not today <= day <= today + SHIPPING_WINDOW_DAYS:
            raise ToolError(
                "RULE_VIOLATION",
                f"day must be between {today} and {today + SHIPPING_WINDOW_DAYS}",
                today_day=today,
            )
        if order.status == "pending":
            raise ToolError("INVALID_STATE", f"order {order.order_id} is unpaid and cannot ship")
        if order.status != "paid":
            raise ToolError(
                "INVALID_STATE", f"order {order.order_id} is {order.status} and cannot be scheduled"
            )
        if not order_fully_reserved(order):
            raise ToolError("RULE_VIOLATION", "reserve stock for every item on the order first")
        shipment = Shipment(
            shipment_id=self.world.new_shipment_id(),
            order_id=order.order_id,
            carrier=carrier,
            scheduled_day=day,
            status="scheduled",
        )
        for item in order.items:
            self.world.products[item.sku].stock -= item.quantity
        order.reservations = []
        order.status = "shipped"
        order.shipment_id = shipment.shipment_id
        self.world.shipments[shipment.shipment_id] = shipment
        self.world.commit()
        return {
            "shipment_id": shipment.shipment_id,
            "order_id": order.order_id,
            "carrier": carrier,
            "scheduled_day": day,
            "status": "scheduled",
        }

    def create_ticket(
        self, customer_id: str, order_id: str, category: str, priority: str, summary: str
    ) -> str:
        """Open a support ticket for a customer, optionally linked to one of their orders.

        Args:
            customer_id: The customer the ticket is for.
            order_id: An order of that customer, or an empty string for no order.
            category: One of billing, shipping, product, account, other.
            priority: One of low, normal, high.
            summary: One-sentence summary of the issue.
        """
        return self._run(
            "create_ticket",
            self._create_ticket,
            customer_id=customer_id,
            order_id=order_id,
            category=category,
            priority=priority,
            summary=summary,
        )

    def _create_ticket(self, customer_id, order_id, category, priority, summary):
        customer = self._customer(_require_text(customer_id, "customer_id"))
        category = _require_text(category, "category").lower()
        priority = _require_text(priority, "priority").lower()
        summary = _require_text(summary, "summary")
        if category not in TICKET_CATEGORIES:
            raise ToolError("INVALID_ARGUMENT", f"category must be one of {list(TICKET_CATEGORIES)}")
        if priority not in TICKET_PRIORITIES:
            raise ToolError("INVALID_ARGUMENT", f"priority must be one of {list(TICKET_PRIORITIES)}")
        linked = None
        if isinstance(order_id, str) and order_id.strip():
            order = self._order(order_id.strip())
            if order.customer_id != customer.customer_id:
                raise ToolError(
                    "RULE_VIOLATION",
                    f"order {order.order_id} belongs to another customer",
                )
            linked = order.order_id
        ticket = Ticket(
            ticket_id=self.world.new_ticket_id(),
            customer_id=customer.customer_id,
            order_id=linked,
            category=category,
            priority=priority,
            summary=summary,
            status="open",
        )
        self.world.tickets[ticket.ticket_id] = ticket
        self.world.commit()
        return {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "order_id": ticket.order_id,
            "category": category,
            "priority": priority,
            "status": "open",
        }

    def wait(self, seconds: int) -> str:
        """Pause for a number of seconds, for example after a rate-limit response.

        Args:
            seconds: How long to wait, between 1 and 600.
        """
        return self._run("wait", self._wait, seconds=seconds)

    def _wait(self, seconds):
        seconds = _require_int(seconds, "seconds", minimum=1)
        if seconds > MAX_WAIT_SECONDS:
            raise ToolError("INVALID_ARGUMENT", f"seconds must be at most {MAX_WAIT_SECONDS}")
        self.world.clock_seconds += seconds
        return {"waited_seconds": seconds, "clock_seconds": self.world.clock_seconds}

    def finish(self, summary: str) -> str:
        """End the task once every requested change has been made.

        Args:
            summary: One sentence describing what was done.
        """
        return self._run("finish", self._finish, summary=summary)

    def _finish(self, summary):
        self.finish_summary = _require_text(summary, "summary")
        self.finished = True
        return {"status": "finished"}
