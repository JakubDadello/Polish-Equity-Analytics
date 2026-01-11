-- ==================================================================
-- initial_labeling.sql
-- Purpose: Generate investment level labels (low/medium/high) 
--          for companies based on sector-level median financial indicators.
-- Input: raw_data table (imported from raw_data.csv)
-- Output: initial_labeling_data table
-- ==================================================================

-- Step 1: Calculate medians per sector + cumulation period
WITH sector_medians AS (
    SELECT 
        sector,
        cumulation,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY net_income) AS med_net_income,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY net_cash_flow) AS med_net_cash_flow,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roe) AS med_roe,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roa) AS med_roa,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ebitda) AS med_ebitda
    FROM raw_data
    GROUP BY sector, cumulation
),

-- Step 2: Calculate score for each company based on indicators above sector median
scores AS (
    SELECT
        d.id,
        d.sector,
        d.cumulation,
        (CASE WHEN d.net_income    > s.med_net_income    THEN 1 ELSE 0 END) +
        (CASE WHEN d.net_cash_flow > s.med_net_cash_flow THEN 1 ELSE 0 END) +
        (CASE WHEN d.roe           > s.med_roe           THEN 1 ELSE 0 END) +
        (CASE WHEN d.roa           > s.med_roa           THEN 1 ELSE 0 END) +
        (CASE WHEN d.ebitda        > s.med_ebitda        THEN 1 ELSE 0 END) AS score
    FROM raw_data d
    JOIN sector_medians s 
      ON d.sector = s.sector 
     AND d.cumulation = s.cumulation
)

-- Step 3: Assign investment_level based on score
CREATE TABLE initial_labeling_data AS 
SELECT 
    r.*,
    CASE 
        WHEN s.score <= 1 THEN 'low' 
        WHEN s.score <= 3 THEN 'medium' 
        ELSE 'high' 
    END AS investment_level
FROM raw_data r
LEFT JOIN scores s 
ON r.id = s.id;
