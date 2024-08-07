import { Transition } from '@headlessui/react';
import { useRouter } from 'next/router';

import { DeleteAgent } from '@/components/Agents/DeleteAgent';
import { KebabMenu } from '@/components/KebabMenu';
import { CoralLogo, Text, Tooltip } from '@/components/Shared';
import { useContextStore } from '@/context';
import { useRecentAgents } from '@/hooks/agents';
import { getIsTouchDevice } from '@/hooks/breakpoint';
import { useSlugRoutes } from '@/hooks/slugRoutes';
import {
  useAgentsStore,
  useCitationsStore,
  useConversationStore,
  useParamsStore,
  useSettingsStore,
} from '@/stores';
import { cn } from '@/utils';
import { getCohereColor } from '@/utils/getCohereColor';

type Props = {
  isExpanded: boolean;
  name: string;
  isBaseAgent?: boolean;
  id?: string;
};

/**
 * @description This component renders an agent card.
 * It shows the agent's name and a colored icon with the first letter of the agent's name.
 * If the agent is a base agent, it shows the Coral logo instead.
 */
export const AgentCard: React.FC<Props> = ({ name, id, isBaseAgent, isExpanded }) => {
  const isTouchDevice = getIsTouchDevice();
  const { conversationId } = useSlugRoutes();
  const router = useRouter();

  const route = router.asPath;
  const isActive = isBaseAgent
    ? conversationId
      ? route === `/c/${conversationId}`
      : route === '/'
    : conversationId
    ? route === `/a/${id}/c/${conversationId}`
    : route === `/a/${id}`;

  const { open, close } = useContextStore();
  const { removeRecentAgentId } = useRecentAgents();
  const { setEditAgentPanelOpen } = useAgentsStore();
  const { setSettings } = useSettingsStore();
  const { resetConversation } = useConversationStore();
  const { resetCitations } = useCitationsStore();
  const { resetFileParams } = useParamsStore();

  const handleNewChat = () => {
    const url = isBaseAgent ? '/' : id ? `/a/${id}` : '/a';
    router.push(url, undefined, { shallow: true });
    setEditAgentPanelOpen(false);
    resetConversation();
    resetCitations();
    resetFileParams();
  };

  const handleEditAssistant = () => {
    if (id) {
      router.push(`/a/${id}`, undefined, { shallow: true });
      setEditAgentPanelOpen(true);
      setSettings({ isConvListPanelOpen: false });
    }
  };

  const handleDeleteAssistant = async () => {
    if (id) {
      open({
        title: 'Delete assistant',
        content: <DeleteAgent name={name} agentId={id} onClose={close} />,
      });
    }
  };

  const handleHideAssistant = () => {
    if (id) removeRecentAgentId(id);
  };

  return (
    <Tooltip label={name} placement="right" hover={!isExpanded}>
      <div
        onClick={handleNewChat}
        className={cn(
          'group flex w-full items-center justify-between gap-x-2 rounded-lg p-2 transition-colors hover:cursor-pointer hover:bg-mushroom-900/80',
          {
            'bg-mushroom-900/80': isActive,
          }
        )}
      >
        <div
          className={cn(
            'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded duration-300',
            id && getCohereColor(id),
            {
              'bg-mushroom-700': isBaseAgent,
            }
          )}
        >
          {isBaseAgent && <CoralLogo style="secondary" />}
          {!isBaseAgent && (
            <Text className="uppercase text-white" styleAs="p-lg">
              {name[0]}
            </Text>
          )}
        </div>
        <Transition
          as="div"
          show={isExpanded}
          className="flex-grow overflow-x-hidden"
          enter="transition-opacity duration-100 ease-in-out delay-300 duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
        >
          <Text className="truncate">{name}</Text>
        </Transition>
        <Transition
          as="div"
          show={isExpanded && !isBaseAgent}
          enter="transition-opacity duration-100 ease-in-out delay-300 duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
        >
          <KebabMenu
            anchor="right start"
            className={cn('flex', {
              'hidden group-hover:flex': !isTouchDevice,
            })}
            items={[
              {
                label: 'New chat',
                onClick: handleNewChat,
                iconName: 'new-message',
              },
              {
                label: 'Hide assistant',
                onClick: handleHideAssistant,
                iconName: 'hide',
              },
              {
                label: 'Edit assistant',
                onClick: handleEditAssistant,
                iconName: 'edit',
              },
              { label: 'Delete assistant', onClick: handleDeleteAssistant, iconName: 'trash' },
            ]}
          />
        </Transition>
      </div>
    </Tooltip>
  );
};
