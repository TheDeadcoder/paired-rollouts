"""Deterministic back-office world: customers, products, orders, shipments, tickets."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field

ORDER_STATUSES = ("pending", "paid", "shipped", "delivered", "cancelled")
CARRIERS = ("standard", "express")
TICKET_CATEGORIES = ("billing", "shipping", "product", "account", "other")
TICKET_PRIORITIES = ("low", "normal", "high")
CUSTOMER_TIERS = ("standard", "gold")
PAGE_SIZE = 5
SHIPPING_WINDOW_DAYS = 7

FIRST_NAMES = (
    "Ayesha", "Farhan", "Nusrat", "Tanvir", "Sadia", "Rafiq", "Mehnaz", "Imran", "Lamia", "Shakil",
    "Priya", "Arjun", "Meera", "Rohan", "Kavya", "Daniel", "Sophie", "Lucas", "Emma", "Noah",
    "Chen", "Wei", "Yuki", "Haruto", "Amara", "Kwame", "Zainab", "Omar", "Elena", "Mateo",
)
LAST_NAMES = (
    "Rahman", "Hossain", "Ahmed", "Islam", "Chowdhury", "Khan", "Sharma", "Patel", "Singh", "Das",
    "Smith", "Johnson", "Brown", "Garcia", "Martinez", "Nguyen", "Kim", "Sato", "Okafor", "Novak",
)
STREETS = (
    "Green Road", "Lake Avenue", "Station Street", "Mill Lane", "Harbor Drive", "Elm Street",
    "Park Boulevard", "River Road", "Market Street", "Hill Crescent",
)
CITIES = (
    "Dhaka", "Chattogram", "Sylhet", "Rajshahi", "Khulna", "Barishal", "Rangpur", "Mymensingh",
    "Comilla", "Gazipur",
)
PRODUCT_NAMES = (
    "Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Laptop Stand", "Webcam 1080p",
    "Noise-Cancelling Headphones", "Portable SSD 1TB", "Monitor Arm", "Desk Lamp", "Ergonomic Chair",
    "Standing Desk Mat", "Bluetooth Speaker", "Smart Plug", "Power Bank 20000mAh", "HDMI Cable 2m",
    "Wireless Charger", "Fitness Tracker", "E-Reader", "Tablet Stylus", "Travel Adapter",
    "Coffee Grinder", "Electric Kettle", "Air Purifier", "Water Filter Jug", "Yoga Mat",
    "Running Shoes", "Backpack 30L", "Insulated Bottle", "Camping Lantern", "Rain Jacket",
    "Board Game Set", "Puzzle 1000 pieces", "Sketchbook A4", "Watercolor Set", "Desk Organizer",
    "Cable Management Kit", "Phone Tripod", "Ring Light", "Microphone USB", "Laptop Sleeve 14in",
)
CANCEL_REASONS = ("customer request", "duplicate order", "out of stock", "fraud check")
REFUND_REASONS = ("damaged item", "late delivery", "customer request", "wrong item")
TICKET_SUMMARIES = (
    "Package arrived damaged", "Charged twice for one order", "Cannot log in to account",
    "Wrong item delivered", "Delivery is late", "Request invoice copy", "Product stopped working",
)


@dataclass
class Address:
    street: str
    city: str
    postal_code: str


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    phone: str
    address: Address
    tier: str


@dataclass
class Product:
    sku: str
    name: str
    price_cents: int
    stock: int


@dataclass
class OrderItem:
    sku: str
    quantity: int
    unit_price_cents: int


@dataclass
class Refund:
    amount_cents: int
    reason: str


@dataclass
class Reservation:
    sku: str
    quantity: int


@dataclass
class Order:
    order_id: str
    customer_id: str
    items: list[OrderItem]
    status: str
    shipping_address: Address
    placed_on_day: int
    paid_before_cancel: bool = False
    refunds: list[Refund] = field(default_factory=list)
    reservations: list[Reservation] = field(default_factory=list)
    shipment_id: str | None = None


@dataclass
class Shipment:
    shipment_id: str
    order_id: str
    carrier: str
    scheduled_day: int
    status: str


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    order_id: str | None
    category: str
    priority: str
    summary: str
    status: str


def order_total_cents(order: Order) -> int:
    return sum(item.quantity * item.unit_price_cents for item in order.items)


def order_refunded_cents(order: Order) -> int:
    return sum(r.amount_cents for r in order.refunds)


def order_reserved_quantity(order: Order, sku: str) -> int:
    return sum(r.quantity for r in order.reservations if r.sku == sku)


def order_fully_reserved(order: Order) -> bool:
    return all(order_reserved_quantity(order, item.sku) >= item.quantity for item in order.items)


class World:
    """Mutable world state with a version history of snapshots."""

    def __init__(self, seed: int, today_day: int = 100):
        self.seed = seed
        self.today_day = today_day
        self.clock_seconds = 0
        self.customers: dict[str, Customer] = {}
        self.products: dict[str, Product] = {}
        self.orders: dict[str, Order] = {}
        self.shipments: dict[str, Shipment] = {}
        self.tickets: dict[str, Ticket] = {}
        self.version = 0
        self.history: list[dict] = []

    @classmethod
    def generate(
        cls, seed: int, n_customers: int = 40, n_products: int = 30, n_orders: int = 80
    ) -> World:
        rng = random.Random(seed)
        world = cls(seed)
        used_names: set[str] = set()
        for i in range(1, n_customers + 1):
            if i > 1 and rng.random() < 0.3:
                twin = world.customers[f"C{rng.randint(1, i - 1):04d}"]
                last = twin.name.split(" ")[-1]
                first = rng.choice(FIRST_NAMES)
            else:
                first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            name = f"{first} {last}"
            while name in used_names:
                first = rng.choice(FIRST_NAMES)
                name = f"{first} {last}"
            used_names.add(name)
            world.customers[f"C{i:04d}"] = Customer(
                customer_id=f"C{i:04d}",
                name=name,
                email=f"{first.lower()}.{last.lower()}{i}@example.com",
                phone="+880" + "".join(str(rng.randint(0, 9)) for _ in range(10)),
                address=_random_address(rng),
                tier=rng.choice(CUSTOMER_TIERS),
            )
        product_names = list(PRODUCT_NAMES)
        rng.shuffle(product_names)
        for i in range(1, n_products + 1):
            world.products[f"SKU-{i:04d}"] = Product(
                sku=f"SKU-{i:04d}",
                name=product_names[(i - 1) % len(product_names)],
                price_cents=rng.randrange(500, 50001, 50),
                stock=rng.choice([0, 0, 2, 5, 10, 20, 50]),
            )
        skus = sorted(world.products)
        customer_ids = sorted(world.customers)
        for i in range(1, n_orders + 1):
            order_id = f"O{i:04d}"
            customer = world.customers[rng.choice(customer_ids)]
            items = []
            for sku in rng.sample(skus, rng.randint(1, 3)):
                items.append(OrderItem(sku, rng.randint(1, 3), world.products[sku].price_cents))
            status = rng.choices(ORDER_STATUSES, weights=[25, 30, 20, 15, 10])[0]
            order = Order(
                order_id=order_id,
                customer_id=customer.customer_id,
                items=items,
                status=status,
                shipping_address=copy.deepcopy(customer.address)
                if rng.random() < 0.8
                else _random_address(rng),
                placed_on_day=world.today_day - rng.randint(0, 30),
            )
            if status in ("shipped", "delivered"):
                shipment_id = f"S{len(world.shipments) + 1:04d}"
                order.shipment_id = shipment_id
                world.shipments[shipment_id] = Shipment(
                    shipment_id=shipment_id,
                    order_id=order_id,
                    carrier=rng.choice(CARRIERS),
                    scheduled_day=order.placed_on_day + rng.randint(1, 5),
                    status="delivered" if status == "delivered" else "in_transit",
                )
                if status == "delivered" and rng.random() < 0.3:
                    item = rng.choice(items)
                    order.refunds.append(Refund(item.unit_price_cents, rng.choice(REFUND_REASONS)))
            elif status == "paid" and rng.random() < 0.5:
                order.reservations = [Reservation(it.sku, it.quantity) for it in items]
            elif status == "cancelled":
                order.paid_before_cancel = rng.random() < 0.5
                if order.paid_before_cancel and rng.random() < 0.6:
                    order.refunds.append(Refund(order_total_cents(order), "cancelled order"))
            world.orders[order_id] = order
        for sku, product in world.products.items():
            product.stock = max(product.stock, world.reserved_quantity(sku))
        order_ids = sorted(world.orders)
        for i in range(1, 11):
            order = world.orders[rng.choice(order_ids)]
            world.tickets[f"T{i:04d}"] = Ticket(
                ticket_id=f"T{i:04d}",
                customer_id=order.customer_id,
                order_id=order.order_id if rng.random() < 0.7 else None,
                category=rng.choice(TICKET_CATEGORIES),
                priority=rng.choice(TICKET_PRIORITIES),
                summary=rng.choice(TICKET_SUMMARIES),
                status=rng.choice(("open", "closed")),
            )
        world.history = [world.snapshot()]
        return world

    def snapshot(self) -> dict:
        return {
            "today_day": self.today_day,
            "clock_seconds": self.clock_seconds,
            "version": self.version,
            "customers": {k: asdict(v) for k, v in sorted(self.customers.items())},
            "products": {k: asdict(v) for k, v in sorted(self.products.items())},
            "orders": {k: asdict(v) for k, v in sorted(self.orders.items())},
            "shipments": {k: asdict(v) for k, v in sorted(self.shipments.items())},
            "tickets": {k: asdict(v) for k, v in sorted(self.tickets.items())},
        }

    def commit(self) -> int:
        self.version += 1
        self.history.append(self.snapshot())
        return self.version

    def as_of(self, version: int) -> dict:
        version = max(0, min(version, len(self.history) - 1))
        return copy.deepcopy(self.history[version])

    def reserved_quantity(self, sku: str) -> int:
        return sum(order_reserved_quantity(o, sku) for o in self.orders.values())

    def available(self, sku: str) -> int:
        return self.products[sku].stock - self.reserved_quantity(sku)

    def new_shipment_id(self) -> str:
        return f"S{len(self.shipments) + 1:04d}"

    def new_ticket_id(self) -> str:
        return f"T{len(self.tickets) + 1:04d}"


def _random_address(rng: random.Random) -> Address:
    return Address(
        street=f"{rng.randint(1, 250)} {rng.choice(STREETS)}",
        city=rng.choice(CITIES),
        postal_code=f"{rng.randint(1000, 9999)}",
    )
