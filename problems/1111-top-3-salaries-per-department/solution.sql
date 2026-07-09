WITH RankedEmployees AS (
    SELECT 
        department,
        name,
        salary,
        DENSE_RANK() OVER (
            PARTITION BY department 
            ORDER BY salary DESC
        ) AS rnk
    FROM 
        employees
)
SELECT 
    department,
    name,
    salary,
    rnk
FROM 
    RankedEmployees
WHERE 
    rnk <= 3
ORDER BY 
    department ASC,
    salary DESC,
    name ASC;