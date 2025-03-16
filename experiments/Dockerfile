# Use the official Neo4j image from Docker Hub
FROM neo4j:4.2.3

# Set the default password for Neo4j (adjust as needed)
ENV NEO4J_AUTH=neo4j/123412345

# Copy the simkit plugin jar into the plugins directory
# Ensure simkit.jar is in the same directory as this Dockerfile
COPY simkit.jar /var/lib/neo4j/plugins/

# Allow Neo4j to load simkit procedures without restrictions
RUN echo "dbms.security.procedures.unrestricted=simkit.*" >> /var/lib/neo4j/conf/neo4j.conf

# Expose the Bolt and HTTP ports
EXPOSE 7687 7474