-- Number of rows in the database
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1

-- Number of usable rows
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Labelled "test performed"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE test_performed IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Unlabelled "test performed"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE test_performed IS NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Class breakdown for "test performed"
SELECT test_performed, COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE test_performed IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )
GROUP BY test_performed

-- Labelled "test outcome"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE test_outcome IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Unlabelled "test outcome"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE test_outcome IS NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Class breakdown for "test outcome"
SELECT test_outcome, COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE test_outcome IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )
GROUP BY test_outcome

-- Labelled "level 1"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE level_1 IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Unlabelled "level 1"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE level_1 IS NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Class breakdown for "level 1"
SELECT level_1, COUNT(*) AS cnt
FROM lab.dim_test_result_output_v1 AS table_0
WHERE level_1 IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )
GROUP BY level_1
ORDER BY cnt DESC

-- Labelled "level 2"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE level_2 IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Unlabelled "level 2"
SELECT COUNT(*)
FROM lab.dim_test_result_output_v1 AS table_0
WHERE level_2 IS NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )

-- Class breakdown for "level 2"
SELECT level_2, COUNT(*) AS cnt
FROM lab.dim_test_result_output_v1 AS table_0
WHERE level_2 IS NOT NULL
	AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR, lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )
GROUP BY level_2
ORDER BY cnt DESC
