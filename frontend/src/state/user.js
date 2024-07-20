import { proxy } from "valtio";

export const userState = proxy({
  isLoggedIn: false,
  username: "",
  email: "",

  setUser: (userData) => {
    if (userData && userData.is_active) {
      userState.isLoggedIn = true;
      userState.username = userData.username;
      userState.email = userData.email;
    }
  },
});
