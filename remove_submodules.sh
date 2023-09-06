#!/usr/bin/env bash

# The each workbook from the enclave has it's own git repository but it isn't publically available.
# This means we can't add them as submodules. Here we clone the repo then strip out the git
# directory so we can have a monorepo.

find . -mindepth 1 -name .git | xargs rm -rf