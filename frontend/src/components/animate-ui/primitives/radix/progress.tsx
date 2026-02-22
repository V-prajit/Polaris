'use client';

import * as React from 'react';
import * as ProgressPrimitive from '@radix-ui/react-progress';
import { cn } from '@/lib/utils';

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ className, value, ...props }, ref) => (
  <ProgressPrimitive.Root
    ref={ref}
    className={cn('relative w-full overflow-hidden rounded-full', className)}
    {...props}
  >
    {props.children}
  </ProgressPrimitive.Root>
));
Progress.displayName = 'Progress';

const ProgressIndicator = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Indicator>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Indicator> & {
    value?: number;
  }
>(({ className, style, ...props }, ref) => {
  const root = React.useContext(ProgressContext);
  const val = root ?? 0;
  return (
    <ProgressPrimitive.Indicator
      ref={ref}
      className={cn('transition-transform duration-300 ease-out', className)}
      style={{
        transform: `translateX(-${100 - val}%)`,
        ...style,
      }}
      {...props}
    />
  );
});
ProgressIndicator.displayName = 'ProgressIndicator';

const ProgressContext = React.createContext<number>(0);

const ProgressProvider = ({
  value,
  children,
}: {
  value: number;
  children: React.ReactNode;
}) => (
  <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>
);

// Wrap Progress to inject value into context for ProgressIndicator
const ProgressWithContext = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ value, children, ...props }, ref) => (
  <ProgressProvider value={value ?? 0}>
    <Progress ref={ref} value={value} {...props}>
      {children}
    </Progress>
  </ProgressProvider>
));
ProgressWithContext.displayName = 'Progress';

export { ProgressWithContext as Progress, ProgressIndicator };
