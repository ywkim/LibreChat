import { useRecoilValue } from 'recoil';
import { useGetEndpointsQuery } from 'librechat-data-provider';
import type { TConversation, TPreset } from 'librechat-data-provider';
import { getDefaultEndpoint, buildDefaultConvo } from '~/utils';
import store from '~/store';

type TDefaultConvo = { conversation: Partial<TConversation>; preset?: Partial<TPreset> | null };

const useDefaultConvo = () => {
  const { data: endpointsConfig = {} } = useGetEndpointsQuery();
  const modelsConfig = useRecoilValue(store.modelsConfig);

  const getDefaultConversation = ({ conversation, preset }: TDefaultConvo) => {
    const endpoint = getDefaultEndpoint({
      convoSetup: preset as TPreset,
      endpointsConfig,
    });
    const models = modelsConfig?.[endpoint] || [];

    return buildDefaultConvo({
      conversation: conversation as TConversation,
      endpoint,
      lastConversationSetup: preset as TConversation,
      models,
    });
  };

  return getDefaultConversation;
};

export default useDefaultConvo;
