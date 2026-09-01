# A thin layer for iterating on the code.
#
# Everything heavy (conda environment, Node 18, JVM, Minecraft server and world,
# node_modules) lives in the base image. This Dockerfile only copies this
# repository's sources over it, so redeploying edited code pushes a few
# megabytes instead of the whole image.
#
#   docker build -t mineamongus:dev .
#   docker run -it --shm-size=4g mineamongus:dev
#
# To build on a locally loaded image instead of the published one:
#   docker build --build-arg BASE=mineamongus:paper -t mineamongus:dev .
ARG BASE=ghcr.io/junseokim0103/mineamongus:paper
FROM ${BASE}

# COPY merges directories, so mineland/sim/server/ and
# mineland/sim/mineflayer/node_modules/ — which are not tracked in this
# repository — survive from the base image while the sources are updated.
COPY mineland/ /root/MineLand/mineland/
COPY scripts/  /root/MineLand/scripts/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/run_2vs6.sh   /usr/local/bin/run_2vs6.sh

RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/run_2vs6.sh \
 && find /root/MineLand -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

ENV DISPLAY=:1
ENV PATH=/root/anaconda3/envs/mineland/bin:/root/.nvm/versions/node/v18.18.2/bin:$PATH
WORKDIR /root/MineLand
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
