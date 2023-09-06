

@transform_pandas(
    Output(rid="ri.vector.main.execute.ae4ab502-6abb-4cb3-8da9-80bc303f75bc"),
    lag_dx_to_admit=Input(rid="ri.foundry.main.dataset.c2c6f85c-349a-4bb7-97a0-6d4bb9919547")
)
dist_hosp <- function(lag_dx_to_admit) {
    hosplag <- lag_dx_to_admit$hosplag[!is.na(lag_dx_to_admit$hosplag)]
    brks <- seq(floor(min(hosplag)), ceiling(max(hosplag)))
    hist(hosplag, breaks=brks)
    return(lag_dx_to_admit)
}

