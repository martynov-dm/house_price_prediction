import {
  Box,
  Button,
  Flex,
  FormControl,
  FormLabel,
  NumberInput,
  NumberInputField,
  useColorModeValue,
  VStack,
} from "@chakra-ui/react";
import { useForm } from "react-hook-form";
import CommonForm from "./CommonForm";

const RuForm = () => {
  const {
    handleSubmit,
    register,
    reset,
    formState: { errors },
  } = useForm();
  const formBgColor = useColorModeValue("gray.50", "gray.700");
  const inputBgColor = useColorModeValue("white", "gray.600");

  const onSubmit = (data) => {
    console.log(data);
    reset();
  };

  return (
    <Flex my={8} direction="column">
      <Box maxW="800px" mx="auto" w="100%">
        <VStack spacing={8} align="stretch">
          <Box bg={formBgColor} p={6} borderRadius="md" boxShadow="md">
            <form onSubmit={handleSubmit(onSubmit)}>
              <CommonForm register={register} errors={errors} />

              <FormControl mb={4} isInvalid={errors.distance_to_center}>
                <FormLabel>Расстояние до центра города (км)</FormLabel>
                <NumberInput min={0}>
                  <NumberInputField
                    {...register("distance_to_center", {
                      valueAsNumber: true,
                      required: "Это поле обязательно",
                      min: {
                        value: 0,
                        message: "Значение должно быть не менее 0",
                      },
                    })}
                    bg={inputBgColor}
                  />
                </NumberInput>
              </FormControl>

              <Button mt={4} colorScheme="teal" type="submit">
                Предсказать цену
              </Button>
            </form>
          </Box>
        </VStack>
      </Box>
    </Flex>
  );
};

export default RuForm;
