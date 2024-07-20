import axios from "axios";
import getApiUrl from "../utils/getApiUrl";
import.meta.env.PROD;

const axiosClient = axios.create({
  baseURL: getApiUrl(),
  headers: {
    "Content-type": "application/json",
  },
  withCredentials: true,
});

axiosClient.interceptors.response.use(
  function (response) {
    return response;
  },
  function (error) {
    if (error.response.data.detail) {
      const errorMessage = error.response.data.detail;
      const customError = new Error(errorMessage);
      return Promise.reject(customError);
    }
    return Promise.reject(error);
  }
);

export default axiosClient;
