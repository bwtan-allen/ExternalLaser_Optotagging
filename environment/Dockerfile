# hash:sha256:b78697eeb4ad6ebb1f84b82d61635cfd25a9fbebdbe30eb156a463d606b3f199
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install --no-cache-dir \
	aind-codeocean-pipeline-monitor==0.11.2 \
	aind-data-schema==2.9.0 \
	aind-ephys-utils==0.4.0 \
	codeocean==0.13.0 \
	matplotlib==3.11.1 \
	numpy==2.5.1 \
	open-ephys-python-tools==1.0.1 \
	pandas==3.0.5 \
	requests==2.32.3 \
	scipy==1.18.0 \
	spikeinterface==0.104.7 \
	statsmodels==0.14.6 \
	tqdm==4.70.0 \
	zarr==2.18.7
