import { atom } from 'recoil';

const lang = atom({
  key: 'lang',
  default: localStorage.getItem('lang') || 'ko',
});

export default { lang };
