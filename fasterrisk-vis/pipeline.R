

@transform_pandas(
    Output(rid="ri.vector.main.execute.b8613199-a896-418b-bc27-548da0726110"),
    Fr_1a_test_set_eval=Input(rid="ri.foundry.main.dataset.be2d78e5-ecc3-46c7-a73e-0ce5d75921f4")
)
library(ggplot2)
task_1a_calibration_ggplot <- function(Fr_1a_test_set_eval) {
    
    pltdata <- Fr_1a_test_set_eval
    pltdata <- pltdata[pltdata$N > 20, ]

    pltdata$npos <- round(pltdata$N * pltdata$empirical_prob)
    pltdata$nneg <- pltdata$N - pltdata$npos

    pltdata$pmin <- ifelse(pltdata$npos >= 1, as.numeric(qbeta(0.05, pltdata$npos+1, pltdata$nneg+1)), 0)
    pltdata$pmax <- as.numeric(qbeta(0.95, pltdata$npos+1, pltdata$nneg+1))

    plt <- ggplot(pltdata, aes(x=model_prob)) +
             geom_point(mapping=aes(y=empirical_prob), size=2) +
             geom_errorbar(mapping=aes(ymin=pmin, ymax=pmax), width=0.002) +
             geom_abline(slope=1, intercept=0, linetype=2) +
             ggtitle('(A)') +
             xlab('Model risk') + ylab('Empirical risk') +
             theme_bw(20)

    print(plt)

    return(NULL)
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.42850f08-4292-4238-8c9c-d215f5bc0323"),
    Fr_1b_test_set_eval=Input(rid="ri.foundry.main.dataset.95df0b48-eb91-48cd-9477-3855645c2bc4")
)
library(ggplot2)
task_1b_calibration_ggplot <- function(Fr_1b_test_set_eval) {
    
    pltdata <- Fr_1b_test_set_eval
    pltdata <- pltdata[pltdata$N > 20, ]

    pltdata$npos <- round(pltdata$N * pltdata$empirical_prob)
    pltdata$nneg <- pltdata$N - pltdata$npos

    pltdata$pmin <- ifelse(pltdata$npos >= 1, as.numeric(qbeta(0.05, pltdata$npos+1, pltdata$nneg+1)), 0)
    pltdata$pmax <- as.numeric(qbeta(0.95, pltdata$npos+1, pltdata$nneg+1))

    plt <- ggplot(pltdata, aes(x=model_prob)) +
             geom_point(mapping=aes(y=empirical_prob), size=2) +
             geom_errorbar(mapping=aes(ymin=pmin, ymax=pmax), width=0.002) +
             geom_abline(slope=1, intercept=0, linetype=2) +
             ggtitle('(B)') +
             xlab('Model risk') + ylab('Empirical risk') +
             theme_bw(20)

    print(plt)

    return(NULL)
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.d66b30e4-b9cf-4d02-8780-3977c803d326"),
    Fr_1a_test_set_eval=Input(rid="ri.foundry.main.dataset.be2d78e5-ecc3-46c7-a73e-0ce5d75921f4")
)
unnamed <- function(Fr_1a_test_set_eval) {
    
}

