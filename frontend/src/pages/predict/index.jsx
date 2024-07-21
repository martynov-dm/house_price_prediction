import {
  Box,
  FormControl,
  FormLabel,
  Heading,
  Select,
  Text,
  useColorModeValue,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import MskForm from "./components/MskForm";
import RuForm from "./components/RuForm";

const GeneratePage = () => {
  const [selectedRegion, setSelectedRegion] = useState("");
  const [predictedPrice, setPredictedPrice] = useState(null);

  const formBgColor = useColorModeValue("gray.50", "gray.700");
  const inputBgColor = useColorModeValue("white", "gray.600");

  const handleRegionChange = (event) => {
    setSelectedRegion(event.target.value);
    setPredictedPrice(null);
  };

  return (
    <Box maxWidth="800px" margin="auto" padding={8}>
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="xl" textAlign="center">
          Оценка стоимости загородной недвижимости
        </Heading>
        <Box bg={formBgColor} p={6} borderRadius="md">
          <FormControl>
            <FormLabel>Выберите регион</FormLabel>
            <Select
              value={selectedRegion}
              onChange={handleRegionChange}
              bg={inputBgColor}
            >
              <option value="">Выберите регион</option>
              <option value="msk">Москва и Московская область</option>
              <option value="ru">Вся остальная Россия</option>
            </Select>
          </FormControl>
          {selectedRegion === "ru" && (
            <RuForm setPredictedPrice={setPredictedPrice} />
          )}
          {selectedRegion === "msk" && (
            <MskForm setPredictedPrice={setPredictedPrice} />
          )}

          {predictedPrice && (
            <Text fontSize="xl" fontWeight="bold" textAlign="center">
              Предсказанная стоимость: {predictedPrice.toLocaleString()} рублей
            </Text>
          )}
        </Box>
      </VStack>
    </Box>
  );
};

export default GeneratePage;
