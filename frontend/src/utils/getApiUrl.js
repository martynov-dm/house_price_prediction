import isProduction from "./isProduction";

const getApiUrl = () => {
  return isProduction
    ? import.meta.env.VITE_API_PROD_URL
    : import.meta.env.VITE_API_LOCAL_URL;
};

export default getApiUrl;
