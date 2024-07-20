import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import NotFound from "../pages/not_found";
import GeneratePage from "../pages/predict";

const Router = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/predict" />} />
        <Route path="/predict" element={<GeneratePage />} />
        <Route path="/*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
};

export default Router;
