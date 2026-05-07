-- =====================================================
-- CAPACITY PLANNING QUERY
-- =====================================================

WITH Forecast_Base AS
(
    SELECT
        Date,
        Interval,
        Skill,
        Location,
        Forecasted_Volume,
        Forecasted_Hours
    FROM Client_Forecast
),

WFM_Parameters AS
(
    SELECT
        300     AS AHT_SEC,
        0.85    AS OCCUPANCY,
        0.25    AS SHRINKAGE,
        0.80    AS SLA_TARGET
),

Capacity_Calculation AS
(
    SELECT
        F.Date,
        F.Interval,
        F.Skill,
        F.Location,
        F.Forecasted_Volume,
        F.Forecasted_Hours,

        P.AHT_SEC,
        P.OCCUPANCY,
        P.SHRINKAGE,
        P.SLA_TARGET,

        -- Base FTE Requirement
        ROUND(
            (
                F.Forecasted_Volume * P.AHT_SEC
            )
            /
            (
                3600 * P.OCCUPANCY
            ),
            2
        ) AS Base_FTE,

        -- Final Required Headcount
        CEILING(
            (
                (
                    F.Forecasted_Volume * P.AHT_SEC
                )
                /
                (
                    3600 * P.OCCUPANCY
                )
            )
            /
            (
                1 - P.SHRINKAGE
            )
        ) AS Required_Headcount

    FROM Forecast_Base F
    CROSS JOIN WFM_Parameters P
)

-- =====================================================
-- FINAL OUTPUT
-- =====================================================

SELECT
    Date,
    Interval,
    Skill,
    Location,
    Forecasted_Volume,
    Forecasted_Hours,
    AHT_SEC,
    OCCUPANCY,
    SHRINKAGE,
    SLA_TARGET,
    Base_FTE,
    Required_Headcount

FROM Capacity_Calculation

ORDER BY
    Date,
    Interval;
