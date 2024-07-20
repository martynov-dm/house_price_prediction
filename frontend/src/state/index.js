import { proxy } from "valtio";
import { userState } from "./user";

export const store = proxy({
  user: userState,
});
