SELECT 
    plan,
    CAST(DATE_TRUNC('month', signup_date) AS TIMESTAMP) AS cohort_month,
    COUNT(user_id) AS cohort_size,
    ROUND(CAST(SUM(CASE WHEN cancel_date IS NULL OR cancel_date >= signup_date + INTERVAL '1 month' THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(user_id), 4) AS retention_m1,
    ROUND(CAST(SUM(CASE WHEN cancel_date IS NULL OR cancel_date >= signup_date + INTERVAL '2 months' THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(user_id), 4) AS retention_m2,
    ROUND(CAST(SUM(CASE WHEN cancel_date IS NULL OR cancel_date >= signup_date + INTERVAL '3 months' THEN 1 ELSE 0 END) AS NUMERIC) / COUNT(user_id), 4) AS retention_m3
FROM subscriptions
GROUP BY plan, DATE_TRUNC('month', signup_date)
ORDER BY plan, cohort_month;