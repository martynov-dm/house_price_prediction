import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Select,
  useColorModeValue,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import MskForm from "./components/MskForm";
import RuForm from "./components/RuForm";

const GeneratePage = () => {
  const [selectedRegion, setSelectedRegion] = useState("");
  const formBgColor = useColorModeValue("gray.50", "gray.700");
  const inputBgColor = useColorModeValue("white", "gray.600");

  const handleRegionChange = (event) => {
    setSelectedRegion(event.target.value);
  };

  return (
    <Flex my={8} direction="column">
      <Heading textAlign="center" mb={8}>
        Оценка стоимости загородной недвижимости
      </Heading>
      <Box maxW="800px" mx="auto" w="100%">
        <VStack spacing={8} align="stretch">
          <Box bg={formBgColor} p={6} borderRadius="md" boxShadow="md">
            <FormControl id="region" mb={4}>
              <FormLabel>Выберите регион</FormLabel>
              <Select
                placeholder="Выберите регион"
                bg={inputBgColor}
                onChange={handleRegionChange}
                value={selectedRegion}
              >
                <option value="msk">Москва и Московская область</option>
                <option value="ru">Вся остальная Россия</option>
              </Select>
            </FormControl>

            {selectedRegion === "ru" && <RuForm />}
            {selectedRegion === "msk" && <MskForm />}
          </Box>
        </VStack>
      </Box>
    </Flex>
  );
};

export default GeneratePage;
