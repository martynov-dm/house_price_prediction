import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  Input,
  NumberInput,
  NumberInputField,
  Select,
  SimpleGrid,
  useColorModeValue,
} from "@chakra-ui/react";

const WallMaterial = {
  BRICK: "Кирпич",
  WOOD: "Дерево",
  BLOCK: "Блок",
  MONOLITH: "Монолит",
};

const Renovation = {
  COSMETIC: "Косметический",
  EURO: "Евроремонт",
  DESIGN: "Дизайнерский",
  WITHOUT: "Без ремонта",
};

const CommonForm = ({ register }) => {
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

  return (
    <>
      <SimpleGrid columns={[1, 2]} spacing={4}>
        <FormControl mb={4}>
          <FormLabel>Площадь дома (м²)</FormLabel>
          <NumberInput min={0}>
            <NumberInputField
              {...register("house_area", {
                valueAsNumber: true,
                required: true,
              })}
              bg={inputBgColor}
            />
          </NumberInput>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Площадь участка (м²)</FormLabel>
          <NumberInput min={0}>
            <NumberInputField
              {...register("land_area", {
                valueAsNumber: true,
                required: true,
              })}
              bg={inputBgColor}
            />
          </NumberInput>
        </FormControl>
      </SimpleGrid>

      <SimpleGrid columns={[1, 2, 3]} spacing={4}>
        <FormControl mb={4}>
          <FormLabel>Количество санузлов</FormLabel>
          <NumberInput min={0}>
            <NumberInputField
              {...register("bathrooms", {
                valueAsNumber: true,
                required: true,
              })}
              bg={inputBgColor}
            />
          </NumberInput>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Материалы стен</FormLabel>
          <Select
            {...register("wall_material", { required: true })}
            bg={inputBgColor}
          >
            {Object.entries(WallMaterial).map(([key, value]) => (
              <option key={key} value={key}>
                {value}
              </option>
            ))}
          </Select>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Количество этажей</FormLabel>
          <NumberInput min={1}>
            <NumberInputField
              {...register("floors", { valueAsNumber: true, required: true })}
              bg={inputBgColor}
            />
          </NumberInput>
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
        <FormControl mb={4}>
          <FormLabel>Количество комнат</FormLabel>
          <NumberInput min={0}>
            <NumberInputField
              {...register("rooms", { valueAsNumber: true, required: true })}
              bg={inputBgColor}
            />
          </NumberInput>
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Тип ремонта</FormLabel>
          <Select
            {...register("renovation", { required: true })}
            bg={inputBgColor}
          >
            {Object.entries(Renovation).map(([key, value]) => (
              <option key={key} value={key}>
                {value}
              </option>
            ))}
          </Select>
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

      <FormControl mb={4}>
        <FormLabel>Год постройки</FormLabel>
        <NumberInput min={1800} max={new Date().getFullYear()}>
          <NumberInputField
            {...register("construction_year", {
              valueAsNumber: true,
              required: true,
            })}
            bg={inputBgColor}
          />
        </NumberInput>
      </FormControl>

      <SimpleGrid columns={[1, 2, 3, 4]} spacing={4}>
        <CheckboxItem name="has_electricity" label="Наличие электричества" />
        <CheckboxItem name="has_gas" label="Наличие газа" />
        <CheckboxItem name="has_heating" label="Наличие отопления" />
        <CheckboxItem name="has_sewerage" label="Наличие канализации" />
      </SimpleGrid>

      <SimpleGrid columns={[1, 2]} spacing={4}>
        <FormControl mb={4}>
          <FormLabel>Город</FormLabel>
          <Input {...register("city", { required: true })} bg={inputBgColor} />
        </FormControl>

        <FormControl mb={4}>
          <FormLabel>Регион</FormLabel>
          <Input
            {...register("region", { required: true })}
            bg={inputBgColor}
            placeholder="Например: Оренбургская область"
          />
        </FormControl>
      </SimpleGrid>
    </>
  );
};

export default CommonForm;
