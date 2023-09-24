import * as Tabs from '@radix-ui/react-tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui';
import { General } from './SettingsTabs';
import { CogIcon } from '~/components/svg';
import { useEffect, useState } from 'react';
import { cn } from '~/utils/';
import { useRecoilValue } from 'recoil';
import { localize } from '~/localization/Translation';
import store from '~/store';

export default function Settings({ open, onOpenChange }) {
  const [confirmClear, setConfirmClear] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const lang = useRecoilValue(store.lang);

  // check if mobile dynamically and update
  useEffect(() => {
    const checkMobile = () => {
      if (window.innerWidth <= 768) {
        setIsMobile(true);
      } else {
        setIsMobile(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    // If the user clicks in the dialog when confirmClear is true, set it to false
    const handleClick = (e) => {
      if (confirmClear) {
        if (e.target.id === 'clearConvosBtn' || e.target.id === 'clearConvosTxt') {
          return;
        }

        setConfirmClear(false);
      }
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [confirmClear]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={cn('shadow-2xl dark:bg-gray-900 dark:text-white md:w-[680px] ')}>
        <DialogHeader>
          <DialogTitle className="text-lg font-medium leading-6 text-gray-900 dark:text-gray-200">
            {localize(lang, 'com_nav_settings')}
          </DialogTitle>
        </DialogHeader>
        <div className="px-6">
          <Tabs.Root
            defaultValue="general"
            className="flex flex-col gap-6 md:flex-row"
            orientation="vertical"
          >
            <Tabs.List
              aria-label="Settings"
              role="tablist"
              aria-orientation="vertical"
              className={cn(
                '-ml-[8px] flex min-w-[180px] flex-shrink-0 flex-col',
                isMobile && 'flex-row rounded-lg bg-gray-100 p-1 dark:bg-gray-800/30',
              )}
              style={{ outline: 'none' }}
            >
              <Tabs.Trigger
                className={cn(
                  'flex items-center justify-start gap-2 rounded-md px-2 py-1.5 text-sm radix-state-active:bg-gray-800 radix-state-active:text-white',
                  isMobile &&
                    'group flex-1 items-center justify-center text-sm dark:text-gray-500 dark:radix-state-active:text-white',
                )}
                value="general"
              >
                <CogIcon />
                {localize(lang, 'com_nav_setting_general')}
              </Tabs.Trigger>
            </Tabs.List>
            <General />
          </Tabs.Root>
        </div>
      </DialogContent>
    </Dialog>
  );
}
