This repo supports the [N3C PHASTR](https://covid.cd2h.org/phastr) project [Use of the N3C enclave and machine learning to create generalizable algorithms that predict patient outcomes at (a) diagnosis, and (b) time of hospitalization](https://covid.cd2h.org/node/778). 

Each directory represents a code workbook in the enclave and can grouped into three categories:

* **Data Cleaning**: these workbooks have the names `{device,drug,procedure,measurement}_{concepts,prefilter}` and select our cohort and analysis window from the larger enclave files

* **Merge** `cohort_join` and `covid_cohort` build the final analysis dataframe from the subset files

* **Modeling** The remainder of the code workbooks build and evaluate the models