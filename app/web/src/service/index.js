import { $axios } from "../../api/index.js";

export const uploadImage = (data) => {
  return $axios.post("/upload/image", data);
};

export const downloadImage = (file_id) => {
  return $axios.get(`/download/image?file_id=${file_id}`, {
    responseType: 'blob'
  });
};

export const historyImages = () => {
  return $axios.get("/history/images");
};
