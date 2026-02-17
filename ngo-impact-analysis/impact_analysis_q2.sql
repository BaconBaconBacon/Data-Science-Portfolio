-- top_regional_impact_assignments
WITH total_donat AS (
	SELECT
		a.assignment_id,
		a.assignment_name,
		COUNT(d.amount) AS num_total_donations
	FROM
		assignments as a
	JOIN
		donations as d
	ON
		a.assignment_id = d.assignment_id
	GROUP BY
		a.assignment_id
	HAVING
		SUM(d.amount) > 0
),
assignment_regions AS (
	SELECT
		ROW_NUMBER()
			OVER (PARTITION BY a.region ORDER BY a.impact_score DESC) AS rank,
		a.assignment_id,
		a.assignment_name,
		a.impact_score,
		a.region,
		t.num_total_donations
	FROM
		assignments AS a
	INNER JOIN
		total_donat as t
	ON
		a.assignment_id = t.assignment_id
)
SELECT
	assignment_name,
	region,
	impact_score,
	num_total_donations
FROM assignment_regions
WHERE rank = 1;