

def krig(df, buf, new_index, N):
    """Interpolates data using Kriging.
    :param df: Dataframe to interpolate with time as the index.
    :param buf: The bandwidth for the empirical variogram
    :param new_index: The new time index on which to get interpolated points
    :param N: Number of nearest points to consider for interpolation.
    :return point estimates, lags, emprircal variogram, model variogram

    """
    #empirical variogram
    def vario(df, buf):
        fr_i = 0
        to_i = ptp(array(df.index))
        lags = arange(fr_i, to_i, buf)
        df_l = df.shape[0]
        dist = spatial.distance.pdist(array(df.index).reshape(df_l, 1))
        sq_dist = spatial.distance.squareform(dist)
        sv_lag = []
        sv = []
        for k in lags:
            for i in range(df_l):
                for j in range(i + 1, df_l):
                    if (sq_dist[i, j] >= k - buf) and (sq_dist[i, j] <= k + buf):
                        sv_lag.append((df.iloc[i] - df.iloc[j])**2)
            sv.append(sum(sv_lag)/(2*len(sv_lag)))
        return array(sv), lags

    #sill
    def c_f(df, lag, lag_i, sv):
        sill = var(df)
        if sv[lag] == 0:
            return sill
        return sill - sv[lag_i]

    #spherical variogram model
    def sph_m(lags, a, c, nugget):
        sph = []
        for i in range(lags.size):
            if lags[i] <= a:
                sph.append(c*( 1.5*lags[i]/a - 0.5*(lags[i]/a)**3.0) + nugget)
            if lags[i] > a:
                sph.append(c + nugget)
        return sph


    def vario_fit(df, buf):
        sv, lags = vario(df, buf) #empirical variogram
        c = c_f(df, lags[0], 0, sv) #sill - nugget
        nugget = sv[0]
        sill = var(df)
        #Fitting the variogram
        sph_par, sph_cv = optimize.curve_fit(sph_m, lags, sv, p0 = [int(lags.size/2), c, nugget])
        sv_model = sph_m(lags, sph_par[0], sph_par[1], sph_par[2])
        return lags, sv, sv_model, sph_par


    lags, sv, sv_model, sph_par = vario_fit(df, buf)
    mu = array(mean(df))
    coord_df = array([repeat(0, array(df.index).size), array(df.index)]).T
    coord_new_index = array([repeat(0, array(new_index).size), new_index]).T
    dist_mat = spatial.distance.cdist(coord_df, coord_new_index)
    dist_mat = c_[df.index, df, dist_mat]
    int_e = []
    for i in range(len(new_index)):
        dist_mat_1 = dist_mat[dist_mat[:,i+2].argsort()]
        dist_mat_1 = dist_mat_1[:N,:]
        k = sph_m(dist_mat_1[:,i+2], sph_par[0], sph_par[1], sph_par[2])
        k = matrix(k).T
        dist_mat_df = spatial.distance.squareform(spatial.distance.pdist(dist_mat_1[:,0].reshape(N,1)))
        K = array(sph_m(dist_mat_df.ravel(), sph_par[0], sph_par[1], sph_par[2])).reshape(N, N)
        K = matrix(K)
        weights = inv(K)*k
        resid = mat(dist_mat_1[:,1] - mu)
        int_e.append(resid*weights + mu)
    int_e = ravel(int_e)
    return int_e, lags, sv, sv_model
