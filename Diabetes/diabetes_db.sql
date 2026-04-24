SELECT
    readmitted,
    COUNT(*)                                    AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM diabetic_data
GROUP BY readmitted
ORDER BY n DESC;

ALTER TABLE diabetic_data
ADD COLUMN readmitted_30day TINYINT DEFAULT 0;

UPDATE diabetic_data
SET readmitted_30day = CASE WHEN readmitted LIKE '%30%' AND readmitted LIKE '%<%' THEN 1 ELSE 0 END;

#Overall readmission rates:
SELECT
    age,
    COUNT(*)                                             AS total,
    SUM(readmitted_30day)                                AS readmitted_30,
    ROUND(AVG(readmitted_30day) * 100, 2)                AS rate_pct
FROM diabetic_data
GROUP BY age
ORDER BY age;

#30-day readmission rate by age group:
SELECT
    number_inpatient,
    COUNT(*)                              AS n,
    ROUND(AVG(readmitted_30day) * 100, 2) AS readmit_rate_pct
FROM diabetic_data
GROUP BY number_inpatient
ORDER BY number_inpatient;

By number of prior inpatient visits — one of your strongest predictors:

SELECT
    insulin,
    COUNT(*)                              AS n,
    ROUND(AVG(readmitted_30day) * 100, 2) AS readmit_rate_pct
FROM diabetic_data
GROUP BY insulin
ORDER BY readmit_rate_pct DESC;

SELECT
    COALESCE(A1Cresult, 'Not tested')    AS a1c_status,
    COUNT(*)                              AS n,
    ROUND(AVG(readmitted_30day) * 100, 2) AS readmit_rate_pct
FROM diabetic_data
GROUP BY a1c_status
ORDER BY readmit_rate_pct DESC;