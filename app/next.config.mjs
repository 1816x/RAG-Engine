/**
 * The browser calls same-origin `/api/*`; those routes proxy to the Python RAG
 * service so the API key and service URL never reach the client.
 */
const nextConfig = {};

export default nextConfig;
