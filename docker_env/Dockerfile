# Base MATLAB image, extended from the matlab-ref-arch group
FROM xychelsea/anaconda3:latest-gpu

USER 0

# Install the apt dependencies
ENV DEBIAN_FROTNEND="noninteractive" TZ="Etc/UTC"
RUN apt-get update \
    && apt-get install --no-install-recommends -y apt-utils libnss-extrausers vim nano emacs \
    && apt-get clean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install extrausers for adding our user on the fly
RUN sed -i '/^passwd:/ s/$/ extrausers/' /etc/nsswitch.conf \
    && sed -i '/^group:/ s/$/ extrausers/' /etc/nsswitch.conf

# Install the conda environment
COPY zonopy.yml zonopy.yml
RUN conda env update --name base --file zonopy.yml --prune \
    && rm -rf zonopy.yml

USER 1000
