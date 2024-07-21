import axiosClient from "./axios-client";

class PredictService {
  async predictRu(body) {
    return axiosClient.post("/predict/ru", body);
  }

  async predictMsk(body) {
    return axiosClient.post("/predict/msk", body);
  }
}

export default new PredictService();
