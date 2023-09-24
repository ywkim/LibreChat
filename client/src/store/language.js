import { atom } from 'recoil';

const lang = atom({
  key: 'lang',
  default: 'ko',
});

export default { lang };
