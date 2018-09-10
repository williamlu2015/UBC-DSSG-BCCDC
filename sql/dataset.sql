SELECT result_full_description
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
