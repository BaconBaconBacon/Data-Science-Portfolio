SELECT
	b.*,
	CASE
		WHEN
			b.height NOT BETWEEN b.lcl AND b.ucl
		THEN TRUE
		ELSE FALSE
	END as alert
FROM (
	SELECT
		a.*,
		-- UCL/LCL: 3-sigma control limits for subgroup size n=5
		-- Formula: x̄ ± 3·σ/√n, where 3 = z-score for 99.73% confidence, √5 = subgroup size
		a.avg_height + 3 * a.stddev_height / SQRT(5) AS ucl,
		a.avg_height - 3 * a.stddev_height / SQRT(5) AS lcl
	FROM (
		SELECT
			operator,
			ROW_NUMBER() OVER w ,
			height,
			AVG(height) OVER w AS avg_height,
			STDDEV(height) OVER w AS stddev_height
		FROM manufacturing_parts
		WINDOW w AS (
			PARTITION BY operator
			ORDER BY item_no
			ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
		)
	) AS a
	WHERE a.row_number >= 5
) AS b;