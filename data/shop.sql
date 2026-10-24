-- Fictional outdoor-gear shop used as the benchmark's SQL environment.
-- Loaded into an in-memory SQLite database; the agent only gets read access.

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price    REAL NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('shipped', 'delivered', 'cancelled', 'returned'))
);

CREATE TABLE order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

INSERT INTO customers VALUES
    (1,  'Anna Schmidt',  'Germany',        '2025-01-14'),
    (2,  'Lukas Weber',   'Germany',        '2025-02-03'),
    (3,  'Marie Dubois',  'France',         '2025-02-20'),
    (4,  'James Carter',  'United Kingdom', '2025-03-08'),
    (5,  'Sofia Rossi',   'Italy',          '2025-03-22'),
    (6,  'Jonas Becker',  'Germany',        '2025-04-10'),
    (7,  'Emma Wilson',   'United Kingdom', '2025-05-01'),
    (8,  'Lucas Martin',  'France',         '2025-05-19'),
    (9,  'Elena Garcia',  'Spain',          '2025-06-07'),
    (10, 'Noah Fischer',  'Austria',        '2025-06-30');

INSERT INTO products VALUES
    (1, 'Trail Runner Shoes', 'footwear',    129.90),
    (2, 'Hiking Boots Pro',   'footwear',    189.00),
    (3, 'Rain Jacket Lite',   'apparel',      99.50),
    (4, 'Merino Base Layer',  'apparel',      64.00),
    (5, 'Trekking Poles',     'gear',         79.00),
    (6, '2-Person Tent',      'gear',        249.00),
    (7, 'Sleeping Bag -5C',   'gear',        159.00),
    (8, 'Headlamp 400',       'accessories',  34.90);

INSERT INTO orders VALUES
    (1,  1,  '2025-07-02', 'delivered'),
    (2,  3,  '2025-07-05', 'delivered'),
    (3,  2,  '2025-07-11', 'cancelled'),
    (4,  4,  '2025-07-15', 'delivered'),
    (5,  1,  '2025-07-28', 'returned'),
    (6,  5,  '2025-08-03', 'delivered'),
    (7,  6,  '2025-08-09', 'delivered'),
    (8,  7,  '2025-08-14', 'shipped'),
    (9,  3,  '2025-08-21', 'delivered'),
    (10, 9,  '2025-08-30', 'delivered'),
    (11, 2,  '2025-09-04', 'delivered'),
    (12, 8,  '2025-09-12', 'cancelled'),
    (13, 10, '2025-09-18', 'delivered'),
    (14, 4,  '2025-09-25', 'delivered'),
    (15, 1,  '2025-10-01', 'shipped'),
    (16, 6,  '2025-10-06', 'delivered'),
    (17, 5,  '2025-10-10', 'returned'),
    (18, 9,  '2025-10-15', 'shipped'),
    (19, 7,  '2025-10-19', 'delivered'),
    (20, 2,  '2025-10-22', 'shipped');

INSERT INTO order_items VALUES
    (1, 1, 1), (1, 8, 2),
    (2, 3, 1),
    (3, 6, 1),
    (4, 2, 1), (4, 5, 1),
    (5, 4, 2),
    (6, 7, 1), (6, 8, 1),
    (7, 1, 1),
    (8, 6, 1), (8, 7, 2),
    (9, 4, 1), (9, 5, 2),
    (10, 2, 1),
    (11, 3, 1), (11, 8, 1),
    (12, 1, 2),
    (13, 6, 1), (13, 5, 1),
    (14, 8, 3),
    (15, 2, 1), (15, 4, 1),
    (16, 7, 1),
    (17, 3, 2),
    (18, 1, 1), (18, 4, 2),
    (19, 5, 1),
    (20, 6, 1), (20, 8, 1);
