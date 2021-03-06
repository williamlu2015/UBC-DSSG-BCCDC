SELECT test_key, result_key, result_full_description, LOWER(level_1) AS level_1,
    LOWER(test_type) AS test_type
FROM lab.dim_test_result_output_test_type AS table_0
WHERE level_1 IS NOT NULL
    AND level_1 <> '*not in hierarchy'
    AND level_1 <> '*not in scope'
    AND ISNUMERIC(result_full_description) <> 1
    AND NOT EXISTS (
        SELECT DISTINCT DTR.test_key, DTR.result_key
        FROM lab.dim_test_result DTR, lab.brg_result BR,
            lab.dim_result_hub DRH
        WHERE DTR.result_key = BR.result_key
            AND BR.result_hub_key = DRH.result_hub_key
            AND (DRH.result_code = 'PROF'
                OR DRH.result_code = 'PROTR'
                OR DRH.result_description LIKE '%proficiency%')
            AND DTR.test_key = table_0.test_key
            AND DTR.result_key = table_0.result_key
    )
    AND (test_type = 'antibody'
        OR test_type = 'nat/pcr')
