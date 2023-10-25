import { useCallback } from 'react';
import { useSetRecoilState, useResetRecoilState, useRecoilCallback } from 'recoil';
import { useGetEndpointsQuery } from 'librechat-data-provider';
import type {
  TConversation,
  TMessagesAtom,
  TSubmission,
  TPreset,
  TModelsConfig,
} from 'librechat-data-provider';
import { buildDefaultConvo, getDefaultEndpoint } from '~/utils';
import store from '~/store';

const useConversation = () => {
  const setConversation = useSetRecoilState(store.conversation);
  const setMessages = useSetRecoilState<TMessagesAtom>(store.messages);
  const setSubmission = useSetRecoilState<TSubmission | null>(store.submission);
  const resetLatestMessage = useResetRecoilState(store.latestMessage);
  const { data: endpointsConfig = {} } = useGetEndpointsQuery();

  const switchToConversation = useRecoilCallback(
    ({ snapshot }) =>
      async (
        conversation: TConversation,
        messages: TMessagesAtom = null,
        preset: TPreset | null = null,
        modelsData?: TModelsConfig,
      ) => {
        const modelsConfig = modelsData ?? snapshot.getLoadable(store.modelsConfig).contents;
        const { endpoint = null } = conversation;

        if (endpoint === null) {
          const defaultEndpoint = getDefaultEndpoint({
            convoSetup: preset ?? conversation,
            endpointsConfig,
          });

          const models = modelsConfig?.[defaultEndpoint] ?? [];
          conversation = buildDefaultConvo({
            conversation,
            lastConversationSetup: preset as TConversation,
            endpoint: defaultEndpoint,
            models,
          });
        }

        setConversation(conversation);
        setMessages(messages);
        setSubmission({} as TSubmission);
        resetLatestMessage();
      },
    [endpointsConfig],
  );

  const newConversation = useCallback(
    (template = {}, preset?: TPreset, modelsData?: TModelsConfig) => {
      switchToConversation(
        {
          conversationId: 'new',
          title: 'New Chat',
          ...template,
          endpoint: null,
          createdAt: '',
          updatedAt: '',
        },
        [],
        preset,
        modelsData,
      );
    },
    [switchToConversation],
  );

  const searchPlaceholderConversation = useCallback(() => {
    switchToConversation(
      {
        conversationId: 'search',
        title: 'Search',
        endpoint: null,
        createdAt: '',
        updatedAt: '',
      },
      [],
    );
  }, [switchToConversation]);

  return {
    switchToConversation,
    newConversation,
    searchPlaceholderConversation,
  };
};

export default useConversation;
