-- ============================================================
-- Business-question SQL queries run against the churn.db SQLite
-- database (built by src/sql_loader.py). These are the kind of
-- questions a retention/analytics team would actually ask.
-- ============================================================

-- 1. Overall churn rate
SELECT
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    COUNT(*) AS total_customers
FROM customers;

-- 2. Churn rate by contract type — usually the single strongest driver
SELECT
    Contract,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Revenue at risk: total monthly revenue tied to churned customers, by segment
SELECT
    InternetService,
    COUNT(*) AS churned_customers,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk
FROM customers
WHERE Churn = 'Yes'
GROUP BY InternetService
ORDER BY monthly_revenue_at_risk DESC;

-- 4. Tenure cohort analysis — does churn concentrate in early months?
SELECT
    CASE
        WHEN tenure <= 6  THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        ELSE '24+ months'
    END AS tenure_bucket,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY MIN(tenure);

-- 5. Payment method vs churn — flags friction in billing experience
SELECT
    PaymentMethod,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;
