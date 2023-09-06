library(tidyverse)
library(tidymodels)
library(patchwork)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a3d7fdba-dd95-417d-ae8d-52fd0b44268e"),
    clean_df_for_preds=Input(rid="ri.foundry.main.dataset.87ab1dc2-c9e9-43f4-8de9-5f7c4a33d10e")
)
figure_roc <- function(clean_df_for_preds) {

    big_theme <- theme_bw(base_size = 14) +
        theme(legend.position = c(0.75, 0.25),
            legend.background = element_rect(fill = "white", color = "black"),
            legend.text = element_text(size = 10))

    roc_df <- clean_df_for_preds %>%
        mutate(hosp = as_factor(hosp) %>% fct_infreq() %>% fct_rev()) %>%
        roc_curve(hosp, pred)

    panel_1 <- ggplot(roc_df, aes(1 - specificity, sensitivity)) +
        geom_line() + 
        coord_fixed(expand = FALSE) + 
        big_theme +
        labs( x = "1 - Specificity", y = "Sensitivity")

    hist_df <- 
        clean_df_for_preds %>%
        mutate(g = cut_number(pred, n=20)) %>%
        group_by(g) %>%
        summarise(rate = mean(hosp), pred = mean(pred), n = n())

    panel_2 <- 
        ggplot(hist_df, aes(pred, rate)) + 
            geom_point() +
            coord_fixed(expand = FALSE) + 
            big_theme +
            labs(x = "Predicted Probability", y = "Observed Event Rate")

    panel_a <- (panel_1 + panel_2)

    subset_df <- clean_df_for_preds %>%
        filter(sex != "OTHER") %>%
        mutate(sex = str_to_title(sex) %>% as_factor())

    roc_df <- subset_df %>%
        mutate(hosp = as_factor(hosp) %>% fct_infreq() %>% fct_rev()) %>%
        group_by(sex) %>%
        roc_curve(hosp, pred)

    panel_3 <- ggplot(roc_df, aes(1 - specificity, sensitivity, color = sex)) +
        geom_line() + 
        coord_fixed(expand = FALSE) + 
        big_theme +
        labs( x = "1 - Specificity", y = "Sensitivity", color = "Sex")

    hist_df <- 
        subset_df %>%
        mutate(g = cut_number(pred, n=20)) %>%
        group_by(sex, g) %>%
        summarise(rate = mean(hosp), pred = mean(pred), n = n())

    panel_4 <- 
        ggplot(hist_df, aes(pred, rate, color = sex)) + 
            geom_point() +
            coord_fixed(expand = FALSE) + 
            scale_color_discrete(guide=NULL) + 
            big_theme +
            labs(x = "Predicted Probability", y = "Observed Event Rate") 

    panel_b <- (panel_3 + panel_4)

    subset_df <- clean_df_for_preds %>%
        mutate(race = if_else(race == "Black or African American", "Black", race)) %>%
        mutate(race = str_to_title(race) %>% as_factor() %>% fct_lump_prop(0.12))

    roc_df <- subset_df %>%
        mutate(hosp = as_factor(hosp) %>% fct_infreq() %>% fct_rev()) %>%
        group_by(race) %>%
        roc_curve(hosp, pred)

    panel_5 <- ggplot(roc_df, aes(1 - specificity, sensitivity, color = race)) +
        geom_line() + 
        coord_fixed(expand = FALSE) + 
        big_theme +
        labs( x = "1 - Specificity", y = "Sensitivity", color = "Race")

    hist_df <- 
        subset_df %>%
        mutate(g = cut_number(pred, n=20)) %>%
        group_by(race, g) %>%
        summarise(rate = mean(hosp), pred = mean(pred), n = n())

    panel_6 <- 
        ggplot(hist_df, aes(pred, rate, color = race)) + 
            geom_point() +
            scale_color_discrete(guide=NULL) + 
            coord_fixed(expand = FALSE) + 
            big_theme +
            labs(x = "Predicted Probability", y = "Observed Event Rate")

    panel_c <- (panel_5 + panel_6)
    fig <- panel_1 + panel_2 + panel_3 + panel_4 + panel_5 + panel_6 + plot_layout(ncol = 2) + plot_annotation(tag_levels = "A")
    
    plot(fig)
    return(NULL)
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.64f3ce18-ca98-4f70-8d49-a6b44bf88c18"),
    clean_df_for_figure_1b=Input(rid="ri.foundry.main.dataset.3bdca309-692e-4215-9f30-61d0fdc2e651")
)
figure_roc_1b <- function( clean_df_for_figure_1b) {

    big_theme <- theme_bw(base_size = 14) +
        theme(legend.position = c(0.75, 0.25),
            legend.background = element_rect(fill = "white", color = "black"),
            legend.text = element_text(size = 10))

    roc_df <- clean_df_for_figure_1b %>%
        mutate(composite_outcome = as_factor(composite_outcome) %>% fct_infreq() %>% fct_rev()) %>%
        roc_curve(composite_outcome, pred)

    panel_1 <- ggplot(roc_df, aes(1 - specificity, sensitivity)) +
        geom_line() + 
        coord_fixed(expand = FALSE) + 
        big_theme +
        labs( x = "1 - Specificity", y = "Sensitivity")

    hist_df <- 
        clean_df_for_figure_1b %>%
        mutate(g = cut_number(pred, n=20)) %>%
        group_by(g) %>%
        summarise(rate = mean(composite_outcome), pred = mean(pred), n = n())

    panel_2 <- 
        ggplot(hist_df, aes(pred, rate)) + 
            geom_point() +
            coord_fixed(expand = FALSE) + 
            big_theme +
            labs(x = "Predicted Probability", y = "Observed Event Rate")

    panel_a <- (panel_1 + panel_2)

    subset_df <- clean_df_for_figure_1b %>%
        filter(sex != "OTHER") %>%
        mutate(sex = str_to_title(sex) %>% as_factor())

    roc_df <- subset_df %>%
        mutate(composite_outcome = as_factor(composite_outcome) %>% fct_infreq() %>% fct_rev()) %>%
        group_by(sex) %>%
        roc_curve(composite_outcome, pred)

    panel_3 <- ggplot(roc_df, aes(1 - specificity, sensitivity, color = sex)) +
        geom_line() + 
        coord_fixed(expand = FALSE) + 

        big_theme +
        labs( x = "1 - Specificity", y = "Sensitivity", color = "Sex")

    hist_df <- 
        subset_df %>%
        mutate(g = cut_number(pred, n=20)) %>%
        group_by(sex, g) %>%
        summarise(rate = mean(composite_outcome), pred = mean(pred), n = n())

    panel_4 <- 
        ggplot(hist_df, aes(pred, rate, color = sex)) + 
            geom_point() +
            coord_fixed(expand = FALSE) + 
            scale_color_discrete(guide=NULL) + 
            big_theme +
            labs(x = "Predicted Probability", y = "Observed Event Rate") 

    panel_b <- (panel_3 + panel_4)

    subset_df <- clean_df_for_figure_1b %>%
        mutate(race = if_else(race == "Black or African American", "Black", race)) %>%
        mutate(race = str_to_title(race) %>% as_factor() %>% fct_lump_prop(0.12))

    roc_df <- subset_df %>%
        mutate(composite_outcome = as_factor(composite_outcome) %>% fct_infreq() %>% fct_rev()) %>%
        group_by(race) %>%
        roc_curve(composite_outcome, pred)

    panel_5 <- ggplot(roc_df, aes(1 - specificity, sensitivity, color = race)) +
        geom_line() + 
        coord_fixed(expand = FALSE) + 
        big_theme +
        labs( x = "1 - Specificity", y = "Sensitivity", color = "Race")

    hist_df <- 
        subset_df %>%
        mutate(g = cut_number(pred, n=20)) %>%
        group_by(race, g) %>%
        summarise(rate = mean(composite_outcome), pred = mean(pred), n = n())

    panel_6 <- 
        ggplot(hist_df, aes(pred, rate, color = race)) + 
            geom_point() +
            scale_color_discrete(guide=NULL) + 
            coord_fixed(expand = FALSE) + 
            big_theme +
            labs(x = "Predicted Probability", y = "Observed Event Rate")

    panel_c <- (panel_5 + panel_6)
    
    fig <- panel_1 + panel_2 + panel_3 + panel_4 + panel_5 + panel_6 + plot_layout(ncol = 2) + plot_annotation(tag_levels = "A")
    plot(fig)
    return(NULL)
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.bfd4bb6a-3860-4024-8a29-a95cbc48bd16"),
    Final_model_1a_shap_1=Input(rid="ri.foundry.main.dataset.e7468d12-abc8-43bd-86b1-6c1e92d06277")
)
library(ggridges)
figure_shap_1a <- function(Final_model_1a_shap_1) {

    big_theme <- theme_bw(base_size = 14)

    dat <- Final_model_1a_shap_1 %>%
            select(ends_with("shap")) %>%
            pivot_longer(everything())

    feature_dat <- dat %>%
        group_by(name) %>%
        summarise(avg_effect = mean(abs(value))) %>%
        arrange(desc(avg_effect))

    keep_features <- feature_dat %>% head(10) %>% pull(name)

    subset_dat <- Final_model_1a_shap_1 %>%
            sample_frac(0.5)

    # age_shap nvax_shap htn_shap obesity_shap diabun_shap pregnancy_shap
    
    panel_1 <- subset_dat %>%
        mutate(obesity = if_else(obesity == 1, "Present", "Absent") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(age, age_shap, color = obesity)) +
                    geom_point(size = 0.3, alpha = 0.5) +
                    labs(x = "Age, years", y = "SHAP value", color = "Obesity") +
                    big_theme

    panel_2 <- ggplot(subset_dat, aes(nvax, nvax_shap, color = age)) +
                geom_jitter(size = 0.3, alpha = 0.5) +
                labs(x = "Number of Recorded COVID19 Vacciations", y = "SHAP value", color = "Age") +
                big_theme

    panel_3 <- subset_dat %>%
        mutate(htn = if_else(htn == 1, "Pre-Existing Hypertension", "None") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(htn, htn_shap, color = age)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            labs(x = "Hypertension", y = "SHAP value", color = "Age") +
            big_theme

    panel_4 <- subset_dat %>%
        mutate(obesity = if_else(obesity == 1, "Obesity", "None") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(obesity, obesity_shap, color = age)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            labs(x = "Obesity", y = "SHAP value", color = "Age") +
            big_theme

    panel_5 <- subset_dat %>%
        mutate(diabun = if_else(diabun == 1, "Pre-Existing Diabetes", "None") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(diabun, diabun_shap, color = age)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            labs(x = "Diabetes", y = "SHAP value", color = "Age") +
            big_theme

    panel_6 <- subset_dat %>%
        mutate(pregnancy = if_else(pregnancy == 1, "Current Pregnancy", "None") %>% as_factor() %>% fct_infreq()) %>%
        mutate(obesity = if_else(obesity == 1, "Present", "Absent") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(pregnancy, pregnancy_shap, color = obesity)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            labs(x = "Pregnancy", y = "SHAP value", color = "Obesity") +
            big_theme

    plot(panel_1 + panel_2 + panel_3 + panel_4 + panel_5 + panel_6 + plot_layout(ncol = 2) + plot_annotation(tag_level = 'A'))

    return(feature_dat)
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.a69d499b-6f10-4122-a26d-4c6a51303693"),
    Final_model_1b_shap=Input(rid="ri.foundry.main.dataset.608c4b0b-116e-46dc-9b09-de2495ecddc6")
)
library(ggridges)
figure_shap_1a_1 <- function(Final_model_1b_shap) {

    big_theme <- theme_bw(base_size = 14)

    dat <- Final_model_1b_shap %>%
            select(ends_with("shap")) %>%
            pivot_longer(everything())

    feature_dat <- dat %>%
        group_by(name) %>%
        summarise(avg_effect = mean(abs(value))) %>%
        arrange(desc(avg_effect))

    subset_dat <- Final_model_1b_shap %>%
            sample_frac(0.5)

    # bun_shap spo2_shap albumin_shap ast_shap age_shap wbc_shap
    
    panel_1 <- ggplot(subset_dat, aes(bun, bun_shap, color = age)) +
                geom_point(size = 0.3, alpha = 0.5) +
                coord_cartesian(xlim = c(5, 150)) + 
                labs(x = "Highest BUN", y = "SHAP value", color = "Age") +
                big_theme

    panel_2 <- ggplot(subset_dat, aes(spo2, spo2_shap, color = age)) +
                geom_jitter(size = 0.3, alpha = 0.5) +
                coord_cartesian(xlim = c(70, 100)) + 
                labs(x = "Lowest SpO2", y = "SHAP value", color = "Age") +
                big_theme

    panel_3 <- subset_dat %>%
        ggplot(aes(albumin, albumin_shap, color = age)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            coord_cartesian(xlim = c(1.5, 5.5)) + 
            labs(x = "Albumin", y = "SHAP value", color = "Age") +
            big_theme

    panel_4 <- subset_dat %>%
        mutate(obesity = if_else(obesity == 1, "Obesity", "None") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(ast, ast_shap, color = obesity)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            coord_cartesian(xlim = c(10, 400)) + 
            labs(x = "AST", y = "SHAP value", color = "Obesity") +
            big_theme

    panel_5 <- subset_dat %>%
        mutate(obesity = if_else(obesity == 1, "Obesity", "None") %>% as_factor() %>% fct_infreq()) %>%
        ggplot(aes(age, age_shap, color = obesity)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            coord_cartesian(xlim = c(18, 89)) + 
            labs(x = "age", y = "SHAP value", color = "Obesity") +
            big_theme

    panel_6 <- subset_dat %>%
        ggplot(aes(wbc, wbc_shap, color = age)) +
            geom_jitter(size = 0.3, alpha = 0.5) +
            coord_cartesian(xlim = c(0, 15)) + 
            labs(x = "WBC", y = "SHAP value", color = "Age") +
            big_theme

    plot(panel_1 + panel_2 + panel_3 + panel_4 + panel_5 + panel_6 + plot_layout(ncol = 2) + plot_annotation(tag_level = 'A'))

    return(feature_dat)
}

