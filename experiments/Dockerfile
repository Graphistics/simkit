# Use the official Neo4j 4.2.3 image
FROM neo4j:5.26

# Set the authentication (username/password)
ENV NEO4J_AUTH=neo4j/123412345

# Copy your locally built SimKit plugin into the plugins directory
# COPY simkit.jar /var/lib/neo4j/plugins/

# Allow SimKit procedures to run unrestricted
# RUN echo "dbms.security.procedures.unrestricted=simkit.*" \
#    >> /var/lib/neo4j/conf/neo4j.conf

# Expose standard Bolt and HTTP ports
EXPOSE 7687 7474
