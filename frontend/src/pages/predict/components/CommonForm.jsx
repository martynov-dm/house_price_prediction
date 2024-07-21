import {
  Checkbox,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Input,
  NumberInput,
  NumberInputField,
  Select,
  SimpleGrid,
  useColorModeValue,
} from "@chakra-ui/react";

const WallMaterial = {
  brevno: "бревно",
  brus: "брус",
  zhelezo_panels: "железобетонные панели",
  kirpich: "кирпич",
  metall: "металл",
  penoblock: "пеноблоки",
  sandvich_panels: "сэндвич-панели",
  experiment_materials: "экспериментальные материалы",
  gazobloki: "газоблоки",
};

const Renovation = {
  designer: "дизайнерский",
  euro: "евро",
  cosmetic: "косметический",
  needs_renovation: "требует ремонта",
};

const Bathroom = {
  both: "на улице и в доме",
  inside: "в доме",
  outside: "на улице",
};

const CommonForm = ({ register, errors }) => {
  const inputBgColor = useColorModeValue("white", "gray.600");

  const CheckboxItem = ({ name, label }) => (
    <FormControl mb={4}>
      <Flex align="center" justify="flex-start" height="100%">
        <Checkbox textAlign={"left"} {...register(name)}>
          {label}
        </Checkbox>
      </Flex>
    </FormControl>
  );
  //   console.log(JSON.stringify(errors));
  return (
    <>
      <SimpleGrid columns={[1, 2]} spacing={4}>
        <FormControl mb={4} isInvalid={errors.house_area}>
          <FormLabel>Площадь дома (м²)</FormLabel>
          <NumberInput min={20} max={1000}>
            <NumberInputField
              {...register("house_area", {
                valueAsNumber: true,
                required: "Это поле обязательно",
                min: { value: 20, message: "Минимальная площадь дома 20 м²" },
                max: {
                  value: 1000,
                  message: "Максимальная площадь дома 1000 м²",
                },
              })}
              bg={inputBgColor}
            />
          </NumberInput>
          <FormErrorMessage>
            {errors.house_area && errors.house_area.message}
          </FormErrorMessage>
        </FormControl>

        <FormControl mb={4} isInvalid={errors.land_area}>
          <FormLabel>Площадь участка (сот.)</FormLabel>
          <NumberInput min={1} max={500}>
            <NumberInputField
              {...register("land_area", {
                valueAsNumber: true,
                required: "Это поле обязательно",
                min: {
                  value: 1,
                  message: "Минимальная площадь участка 1 сот.",
                },
                max: {
                  value: 500,
                  message: "Максимальная площадь участка 500 сот.",
                },
              })}
              bg={inputBgColor}
            />
          </NumberInput>
          <FormErrorMessage>
            {errors.land_area && errors.land_area.message}
          </FormErrorMessage>
        </FormControl>
      </SimpleGrid>

      <SimpleGrid columns={[1, 2, 3]} spacing={4}>
        <FormControl mb={4} isInvalid={errors.bathroom}>
          <FormLabel>Санузел</FormLabel>
          <Select
            {...register("bathroom", { required: "Выберите тип санузла" })}
            bg={inputBgColor}
          >
            <option value="">Выберите тип санузла</option>
            {Object.entries(Bathroom).map(([key, value]) => (
              <option key={key} value={value}>
                {value.charAt(0).toUpperCase() + value.slice(1)}
              </option>
            ))}
          </Select>
          <FormErrorMessage>
            {errors.bathroom && errors.bathroom.message}
          </FormErrorMessage>
        </FormControl>

        <FormControl mb={4} isInvalid={errors.wall_material}>
          <FormLabel>Материалы стен</FormLabel>
          <Select
            {...register("wall_material", {
              required: "Выберите материал стен",
            })}
            bg={inputBgColor}
          >
            <option value="">Выберите материал стен</option>
            {Object.entries(WallMaterial).map(([key, value]) => (
              <option key={key} value={value}>
                {value.charAt(0).toUpperCase() + value.slice(1)}
              </option>
            ))}
          </Select>
          <FormErrorMessage>
            {errors.wall_material && errors.wall_material.message}
          </FormErrorMessage>
        </FormControl>

        <FormControl mb={4} isInvalid={errors.floors}>
          <FormLabel>Количество этажей</FormLabel>
          <NumberInput min={1} max={5}>
            <NumberInputField
              {...register("floors", {
                valueAsNumber: true,
                required: "Это поле обязательно",
                min: { value: 1, message: "Минимум 1 этаж" },
                max: { value: 5, message: "Максимум 5 этажей" },
              })}
              bg={inputBgColor}
            />
          </NumberInput>
          <FormErrorMessage>
            {errors.floors && errors.floors.message}
          </FormErrorMessage>
        </FormControl>
      </SimpleGrid>

      <SimpleGrid columns={[1, 2, 3]} spacing={4}>
        <CheckboxItem name="has_sauna" label="Наличие бани" />
        <CheckboxItem name="has_pool" label="Наличие бассейна" />
        <CheckboxItem name="has_shop" label="Наличие магазина поблизости" />
        <CheckboxItem name="has_pharmacy" label="Наличие аптеки поблизости" />
        <CheckboxItem
          name="has_kindergarten"
          label="Наличие детского сада поблизости"
        />
        <CheckboxItem name="has_school" label="Наличие школы поблизости" />
        <CheckboxItem name="has_wifi" label="Наличие Wi-Fi" />
        <CheckboxItem name="has_tv" label="Наличие ТВ" />
      </SimpleGrid>

      <SimpleGrid columns={[1, 2]} spacing={4}>
        <FormControl mb={4} isInvalid={errors.rooms}>
          <FormLabel>Количество комнат</FormLabel>
          <NumberInput min={0} max={20}>
            <NumberInputField
              {...register("rooms", {
                valueAsNumber: true,
                required: "Это поле обязательно",
                max: { value: 20, message: "Максимум 20 комнат" },
              })}
              bg={inputBgColor}
            />
          </NumberInput>
          <FormErrorMessage>
            {errors.rooms && errors.rooms.message}
          </FormErrorMessage>
        </FormControl>

        <FormControl mb={4} isInvalid={errors.renovation}>
          <FormLabel>Тип ремонта</FormLabel>
          <Select
            {...register("renovation", { required: "Выберите тип ремонта" })}
            bg={inputBgColor}
          >
            <option value="">Выберите тип ремонта</option>
            {Object.entries(Renovation).map(([key, value]) => (
              <option key={key} value={value}>
                {value.charAt(0).toUpperCase() + value.slice(1)}
              </option>
            ))}
          </Select>
          <FormErrorMessage>
            {errors.renovation && errors.renovation.message}
          </FormErrorMessage>
        </FormControl>
      </SimpleGrid>

      <SimpleGrid columns={[1, 2, 3]} spacing={4}>
        <CheckboxItem
          name="has_open_plan"
          label="Наличие свободной планировки"
        />
        <CheckboxItem name="has_parking" label="Наличие парковки" />
        <CheckboxItem name="has_garage" label="Наличие гаража" />
        <CheckboxItem name="mortgage_available" label="Возможность ипотеки" />
        <CheckboxItem name="has_terrace" label="Наличие террасы" />
        <CheckboxItem
          name="has_asphalt"
          label="Наличие асфальтированной дороги"
        />
        <CheckboxItem
          name="has_public_transport"
          label="Наличие общественного транспорта"
        />
        <CheckboxItem
          name="has_railway"
          label="Наличие железнодорожного сообщения"
        />
      </SimpleGrid>

      <FormControl mb={4} isInvalid={errors.construction_year}>
        <FormLabel>Год постройки</FormLabel>
        <NumberInput min={1800} max={new Date().getFullYear()}>
          <NumberInputField
            {...register("construction_year", {
              valueAsNumber: true,
              required: "Это поле обязательно",
              min: { value: 1800, message: "Минимальный год постройки 1800" },
              max: {
                value: new Date().getFullYear(),
                message: `Максимальный год постройки ${new Date().getFullYear()}`,
              },
            })}
            bg={inputBgColor}
          />
        </NumberInput>
        <FormErrorMessage>
          {errors.construction_year && errors.construction_year.message}
        </FormErrorMessage>
      </FormControl>

      <SimpleGrid columns={[1, 2, 3, 4]} spacing={4}>
        <CheckboxItem name="has_electricity" label="Наличие электричества" />
        <CheckboxItem name="has_gas" label="Наличие газа" />
        <CheckboxItem name="has_heating" label="Наличие отопления" />
        <CheckboxItem name="has_sewerage" label="Наличие канализации" />
      </SimpleGrid>

      <FormControl mb={4} isInvalid={errors.city}>
        <FormLabel>Город</FormLabel>
        <Input
          {...register("city", {
            required: "Укажите город",
            minLength: { value: 2, message: "Минимальная длина 2 символа" },
            maxLength: { value: 50, message: "Максимальная длина 50 символов" },
          })}
          bg={inputBgColor}
        />
        <FormErrorMessage>
          {errors.city && errors.city.message}
        </FormErrorMessage>
      </FormControl>
    </>
  );
};

export default CommonForm;
