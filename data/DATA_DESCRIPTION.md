### Uniquely Named Columns

OK risk:self-computed value at the epoch of each CDM [base 10 log]. In the test set, this value is to be predicted, at the time of closest approach for each event_id. Note that, as mentioned above, in the test set, we do not know the actual data contained in CDMs that are within 2 days to closest approach, since they happen in the "future".

OK event_id: unique id per collision event
time_to_tca: Time interval between CDM creation and time-of-closest approach [days]
OK mission_id: identifier of mission that will be affected

OK max_risk_estimate: maximum collision probability obtained by scaling combined covariance
OK max_risk_scaling: scaling factor used to compute maximum collision probability
OK miss_distance: relative position between chaser & target at tca [m]

OK relative_speed: relative speed between chaser & target at tca [m/s]
OK relative_position_n: relative position between chaser & target: normal (cross-track) [m]
OK relative_position_r: relative position between chaser & target: radial [m]
OK relative_position_t: relative position between chaser & target: transverse (along-track) [m]
OK relative_velocity_n: relative velocity between chaser & target: normal (cross-track) [m/s]
OK relative_velocity_r: relative velocity between chaser & target: radial [m/s]
OK relative_velocity_t: relative velocity between chaser & target: transverse (along-track) [m/s]
OK c_object_type: object type which is at collision risk with satellite

geocentric_latitude: Latitude of conjunction point [deg]
azimuth: relative velocity vector: azimuth angle [deg]
elevation: relative velocity vector: elevation angle [deg]
F10: 10.7 cm radio flux index [10−2210−22 W/(m2m2 Hz)]
AP: daily planetary geomagnetic amplitude index
F3M: 81-day running mean of F10.7 (over 3 solar rotations) [10−2210−22 W/(m2m2 Hz)]
SSN: Wolf sunspot number

### Shared Column Names Between the Chaser and the Target Object

x_sigma_rdot: covariance; radial velocity standard deviation (sigma) [m/s]
x_sigma_n: covariance; (cross-track) position standard deviation (sigma) [m]
x_cn_r: covariance; correlation of normal (cross-track) position vs radial position
x_cn_t: covariance; correlation of normal (cross-track) position vs transverse (along-track) position
x_cndot_n: covariance; correlation of normal (cross-track) velocity vs normal (cross-track) position
x_sigma_ndot: covariance; normal (cross-track) velocity standard deviation (sigma) [m/s]
x_cndot_r: covariance; correlation of normal (cross-track) velocity vs radial position
x_cndot_rdot: covariance; correlation of normal (cross-track) velocity vs radial velocity
x_cndot_t: covariance; correlation of normal (cross-track) velocity vs transverse (along-track) position
x_cndot_tdot: covariance; correlation of normal (cross-track) velocity vs transverse (along-track) velocity
x_sigma_r: covariance; radial position standard deviation (sigma) [m]
x_ct_r: covariance; correlation of transverse (along-track) position vs radial position
x_sigma_t: covariance; transverse (along-track) position standard deviation (sigma) [m]
x_ctdot_n: covariance; correlation of transverse (along-track) velocity vs normal (cross-track) position
x_crdot_n: covariance; correlation of radial velocity vs normal (cross-track) position
x_crdot_t: covariance; correlation of radial velocity vs transverse (along-track) position
x_crdot_r: covariance; correlation of radial velocity vs radial position
x_ctdot_r: covariance; correlation of transverse (along-track) velocity vs radial position
x_ctdot_rdot: covariance; correlation of transverse (along-track) velocity vs radial velocity
x_ctdot_t: covariance; correlation of transverse (along-track) velocity vs transverse (along-track) position
x_sigma_tdot: covariance; transverse (along-track) velocity standard deviation (sigma) [m/s]
x_position_covariance_det: determinant of covariance (~volume)

OK x_cd_area_over_mass: ballistic coefficient [m2m2/kg]
OK x_cr_area_over_mass: solar radiation coefficient . A/m (ballistic coefficient equivalent)
OK x_h_apo: apogee (-RearthRearth) [km]
OK x_h_per: perigee (-RearthRearth)[km]
OK x_ecc: eccentricity
OK x_j2k_inc: inclination [deg]
OK x_j2k_sma: semi-major axis [km]
OK x_sedr: energy dissipation rate [W/kg]
OK x_span: size used by the collision risk computation algorithm (minimum 2 m diameter assumed for the chaser) [m]
OK x_rcs_estimate: radar cross-sectional area [m2m2]
OK x_actual_od_span: actual length of update interval for orbit determination [days]
OK x_obs_available: number of observations available for orbit determination (per CDM)
OK x_obs_used: number of observations used for orbit determination (per CDM)
OK x_recommended_od_span: recommended length of update interval for orbit determination [days]
OK x_residuals_accepted: orbit determination residuals
OK x_time_lastob_end: end of the time interval in days (with respect to the CDM creation epoch) of the last accepted observation used in the orbit determination
OK x_time_lastob_start: start of the time in days (with respect to the CDM creation epoch) of the last accepted observation used in the orbit determination
OK x_weighted_rms: root-mean-square in least-squares orbit determination
