WITH donat AS (
	SELECT
		dona.assignment_id as assignment_id,
		ROUND(SUM(dona.amount),2) AS rounded_amount,
		dono.donor_type as donor_type
	FROM
		donations as dona
	JOIN
		donors AS dono
	ON
		dona.donor_id = dono.donor_id
	GROUP BY
		dona.assignment_id, dono.donor_type
)

SELECT
	a.assignment_name,
	a.region,
	d.rounded_amount as rounded_total_donation_amount,
	d.donor_type as donor_type
FROM
	assignments as a
JOIN
	donat as d
ON
	a.assignment_id = d.assignment_id
ORDER BY
	d.rounded_amount DESC
LIMIT 5;