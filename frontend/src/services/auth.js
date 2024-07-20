import axiosClient from "./axios-client";

class AuthService {
  async register(body) {
    return axiosClient.post("/auth/register", body);
  }

  async login(body) {
    const formData = new FormData();
    formData.append("username", body.email);
    formData.append("password", body.password);

    return axiosClient.post("/auth/jwt/login", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  async me() {
    return axiosClient.get("/users/me");
  }
}

export default new AuthService();
