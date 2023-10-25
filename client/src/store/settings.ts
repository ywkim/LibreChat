import { atom } from 'recoil';

type TOptionSettings = {
  showExamples?: boolean;
  isCodeChat?: boolean;
};

const abortScroll = atom<boolean>({
  key: 'abortScroll',
  default: false,
});

const optionSettings = atom<TOptionSettings>({
  key: 'optionSettings',
  default: {},
});

const showPluginStoreDialog = atom<boolean>({
  key: 'showPluginStoreDialog',
  default: false,
});

const showAgentSettings = atom<boolean>({
  key: 'showAgentSettings',
  default: false,
});

const showBingToneSetting = atom<boolean>({
  key: 'showBingToneSetting',
  default: false,
});

const showPopover = atom<boolean>({
  key: 'showPopover',
  default: false,
});

const autoScroll = atom<boolean>({
  key: 'autoScroll',
  default: localStorage.getItem('autoScroll') === 'true',
  effects: [
    ({ setSelf, onSet }) => {
      const savedValue = localStorage.getItem('autoScroll');
      if (savedValue != null) {
        setSelf(savedValue === 'true');
      }

      onSet((newValue: unknown) => {
        if (typeof newValue === 'boolean') {
          localStorage.setItem('autoScroll', newValue.toString());
        }
      });
    },
  ] as const,
});

export default {
  abortScroll,
  optionSettings,
  showPluginStoreDialog,
  showAgentSettings,
  showBingToneSetting,
  showPopover,
  autoScroll,
};
