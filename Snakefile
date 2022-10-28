rule download_Vasiliev2019:
    output:
        "src/data/Vasiliev2019.ecsv"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/download_Vasiliev2019_table.py"


rule download_IbataEtAl2017:
    output:
        "src/data/IbataEtAl2017.ecsv"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/download_IbataEtAl2017_table.py"


rule download_PWB18_table:
    input:
        "src/data/gd1-with-masks.fits"
    output:
        "src/data/PWB18_thinsel.ecsv"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/download_PWB18_table.py"


rule download_StarkmanEtAl19_table:
    output:
        "src/data/StarkmanEtAl19.ecsv"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/download_StarkmanEtAl19_table.py"


rule simulate_solar_circle:
    output:
        "src/data/solar_circle.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/simulate_solar_circle.py"

rule simulate_streamdf:
    input:
        "src/data/Vasiliev2019.ecsv"
    output:
        "src/data/streamdf.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/simulate_streamdf.py"


rule simulate_streamspraydf:
    input:
        "src/data/Vasiliev2019.ecsv"
    output:
        "src/data/streamspraydf.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/simulate_streamspraydf.py"
