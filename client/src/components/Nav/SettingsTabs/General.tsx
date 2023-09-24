import * as Tabs from '@radix-ui/react-tabs';
import { CheckIcon } from 'lucide-react';
import { DialogButton } from '~/components/ui';
import React, { useState, useContext, useEffect, useCallback } from 'react';
import { useClearConversationsMutation } from 'librechat-data-provider';
import { useRecoilValue, useRecoilState } from 'recoil';
import store from '~/store';
import { ThemeContext } from '~/hooks';
import { cn } from '~/utils';
import { localize } from '~/localization/Translation';

export const ThemeSelector = ({
  theme,
  onChange,
}: {
  theme: string;
  onChange: (value: string) => void;
}) => {
  const lang = useRecoilValue(store.lang);

  return (
    <div className="flex items-center justify-between">
      <div>{localize(lang, 'com_nav_theme')}</div>
      <select
        className="w-24 rounded border border-black/10 bg-transparent text-sm dark:border-white/20 dark:bg-gray-900"
        onChange={(e) => onChange(e.target.value)}
        value={theme}
      >
        <option value="system">{localize(lang, 'com_nav_theme_system')}</option>
        <option value="dark">{localize(lang, 'com_nav_theme_dark')}</option>
        <option value="light">{localize(lang, 'com_nav_theme_light')}</option>
      </select>
    </div>
  );
};

export const ClearChatsButton = ({
  confirmClear,
  className = '',
  showText = true,
  onClick,
}: {
  confirmClear: boolean;
  className?: string;
  showText: boolean;
  onClick: () => void;
}) => {
  const lang = useRecoilValue(store.lang);

  return (
    <div className="flex items-center justify-between">
      {showText && <div>{localize(lang, 'com_nav_clear_all_chats')}</div>}
      <DialogButton
        id="clearConvosBtn"
        onClick={onClick}
        className={cn(
          ' btn btn-danger relative border-none bg-red-700 text-white hover:bg-red-800 dark:hover:bg-red-800',
          className,
        )}
      >
        {/* <button
        className="btn mt-2 inline-flex h-10 items-center justify-center rounded-md relative bg-red-600 text-white hover:bg-red-800"
        type="button"
        id="clearConvosBtn"
        onClick={onClick}
      > */}
        {confirmClear ? (
          <div
            className="flex w-full items-center justify-center gap-2"
            id="clearConvosTxt"
            data-testid="clear-convos-confirm"
          >
            <CheckIcon className="h-5 w-5" /> {localize(lang, 'com_nav_confirm_clear')}
          </div>
        ) : (
          <div
            className="flex w-full items-center justify-center gap-2"
            id="clearConvosTxt"
            data-testid="clear-convos-initial"
          >
            {localize(lang, 'com_nav_clear')}
          </div>
        )}

        {/* </button> */}
      </DialogButton>
    </div>
  );
};

export const LangSelector = ({
  langcode,
  onChange,
}: {
  langcode: string;
  onChange: (value: string) => void;
}) => {
  const lang = useRecoilValue(store.lang);

  return (
    <div className="flex items-center justify-between">
      <div>{localize(lang, 'com_nav_language')}</div>
      <select
        className="w-24 rounded border border-black/10 bg-transparent text-sm dark:border-white/20 dark:bg-gray-900"
        onChange={(e) => onChange(e.target.value)}
        value={langcode}
      >
        <option value="en">{localize(lang, 'com_nav_lang_english')}</option>
        <option value="cn">{localize(lang, 'com_nav_lang_chinese')}</option>
        <option value="de">{localize(lang, 'com_nav_lang_german')}</option>
        <option value="es">{localize(lang, 'com_nav_lang_spanish')}</option>
        <option value="fr">{localize(lang, 'com_nav_lang_french')}</option>
        <option value="it">{localize(lang, 'com_nav_lang_italian')}</option>
        <option value="br">{localize(lang, 'com_nav_lang_brazilian_portuguese')}</option>
      </select>
    </div>
  );
};

function General() {
  const { theme, setTheme } = useContext(ThemeContext);
  const clearConvosMutation = useClearConversationsMutation();
  const [confirmClear, setConfirmClear] = useState(false);
  const [langcode, setLangcode] = useRecoilState(store.lang);
  const { newConversation } = store.useConversation();
  const { refreshConversations } = store.useConversations();

  useEffect(() => {
    if (clearConvosMutation.isSuccess) {
      newConversation();
      refreshConversations();
    }
  }, [clearConvosMutation.isSuccess, newConversation, refreshConversations]);

  const clearConvos = useCallback(() => {
    if (confirmClear) {
      console.log('Clearing conversations...');
      clearConvosMutation.mutate({});
      setConfirmClear(false);
    } else {
      setConfirmClear(true);
    }
  }, [confirmClear, clearConvosMutation]);

  const changeTheme = useCallback(
    (value: string) => {
      setTheme(value);
    },
    [setTheme],
  );

  const changeLang = useCallback(
    (value: string) => {
      setLangcode(value);
    },
    [setLangcode],
  );

  return (
    <Tabs.Content value="general" role="tabpanel" className="w-full md:min-h-[300px]">
      <div className="flex flex-col gap-3 text-sm text-gray-600 dark:text-gray-300">
        <div className="border-b pb-3 last-of-type:border-b-0 dark:border-gray-700">
          <ThemeSelector theme={theme} onChange={changeTheme} />
        </div>
        <div className="border-b pb-3 last-of-type:border-b-0 dark:border-gray-700">
          <LangSelector langcode={langcode} onChange={changeLang} />
        </div>
        <div className="border-b pb-3 last-of-type:border-b-0 dark:border-gray-700">
          <ClearChatsButton confirmClear={confirmClear} onClick={clearConvos} showText={true} />
        </div>
      </div>
    </Tabs.Content>
  );
}

export default React.memo(General);
