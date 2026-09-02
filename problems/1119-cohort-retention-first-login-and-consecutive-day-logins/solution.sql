SELECT 
    user_id, 
    MIN(login_date) AS first_login
FROM (
    SELECT 
        user_id, 
        login_date,
        LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) AS next_login
    FROM logins
) t
GROUP BY user_id
HAVING MAX(CASE WHEN next_login = login_date + INTERVAL '1 day' THEN 1 ELSE 0 END) = 1
ORDER BY user_id ASC;