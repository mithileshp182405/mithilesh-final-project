-- Instamart Customer Intelligence Copilot -- Part A SQL analysis (SQLite 3.25+, database: data/instamart.db)
-- Every query is introduced by a "-- name:" line so that src/a4_sql.py can run them one by one.
-- Only completed orders count as revenue / spend. All amounts are in INR.

-- name: Q1_where_then_having
-- WHERE filters individual orders BEFORE grouping; HAVING filters whole customer groups AFTER aggregation.
SELECT customer_id,
       COUNT(*)                     AS n_big_orders,
       ROUND(SUM(order_amount), 2)  AS total_big_order_inr
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_big_order_inr DESC;

-- name: Q2_revenue_by_city_and_category
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders    AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products  AS p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;

-- name: Q3_products_above_average_price
-- Scalar subquery: returns one number (the catalogue average) that the outer WHERE compares against.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- name: Q4_customers_without_completed_order
-- The status test sits in the ON clause, not in WHERE: a WHERE test would remove the NULL rows we are looking for
-- and silently turn the LEFT JOIN into an inner join.
SELECT c.customer_id, c.city, c.signup_date
FROM customers AS c
LEFT JOIN orders AS o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5_cte_above_average_spenders
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT cs.customer_id,
       ROUND(cs.total_spend, 2)      AS total_spend_inr,
       ROUND(a.avg_total_spend, 2)   AS avg_total_spend_inr
FROM customer_spend AS cs
CROSS JOIN average_spend AS a
WHERE cs.total_spend > a.avg_total_spend
ORDER BY cs.total_spend DESC, cs.customer_id;

-- name: Q6_rank_within_city
-- Spend is rounded to paise first so that floating-point noise cannot split a genuine tie.
WITH customer_spend AS (
    SELECT c.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders    AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT city,
       customer_id,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;

-- name: Q7_daily_revenue_running_total
-- The day is the first ten characters of order_ts. Days are unique after the GROUP BY, so the default window frame
-- (RANGE UNBOUNDED PRECEDING .. CURRENT ROW) is exactly a running total.
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day,
           SUM(order_amount)       AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2)                            AS revenue_inr,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)   AS cumulative_revenue_inr
FROM daily
ORDER BY day;
