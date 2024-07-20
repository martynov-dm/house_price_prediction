import { useQuery } from "@tanstack/react-query";
import authService from "../services/auth";

const useAuth = () => {
  const { isLoading, isError, refetch, data } = useQuery({
    queryKey: ["userData"],
    queryFn: authService.me,
    retry: 0,
    staleTime: 1000 * 60 * 60 * 4, // 4 hours
  });
  return {
    user: data?.data,
    isLoading,
    isError,
    refetch,
  };
};

export default useAuth;
